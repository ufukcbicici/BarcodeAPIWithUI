package bandrol_training.post_processing;

import bandrol_training.Constants;
import bandrol_training.model.DbUtils;
import bandrol_training.model.Detection;
import bandrol_training.model.GroundTruth;
import bandrol_training.model.Utils;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.util.Pair;
import org.opencv.core.*;
import org.opencv.ml.EM;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import java.util.*;
import java.util.stream.Collectors;

import static bandrol_training.Constants.TOTAL_CHAR_COUNT;

class OLSResult
{
    private Mat weights;
    private double sumOfSquaredResiduals;
    private double variance;

    public OLSResult(Mat weights, double variance, double sumOfSquaredResiduals)
    {
        this.weights = weights;
        this.variance = variance;
        this.sumOfSquaredResiduals = sumOfSquaredResiduals;
    }

    public Mat getWeights() {
        return weights;
    }

    public double getVariance() {
        return variance;
    }

    public double getSumOfSquaredResiduals()
    {
        return getSumOfSquaredResiduals();
    }
};

public class ProbabilisticPostProcesser {

    private int topLeftMixtureCount;
    private double [] gaussianPriors;
    private MultivariateNormalDistribution [] topLeftGaussianComponents;
    private Mat horizontalGaussianCov;
    private Mat inverseHorizontalGaussianCov;
    private Mat verticalGaussianCov;
    private Mat inverseVerticalGaussianCov;
    private OLSResult horizontalXRes;
    //System.out.println(horizontalXRes.getWeights().dump());
    private OLSResult horizontalYRes;
    //System.out.println(horizontalYRes.getWeights().dump());
    private OLSResult verticalXRes;
    //System.out.println(verticalXRes.getWeights().dump());
    private OLSResult verticalYRes;

    public void train(int topLeftMixtureCount)
    {
        this.topLeftMixtureCount = topLeftMixtureCount;
        // Read all ground truth bounding boxes.
        String exclusionStatement = "FileName NOT IN" + Utils.getFileSelectionClause();
        String filterClause = Utils.getFilterClause(
                "Label != -1",
                "ABS(VerticalDisplacement) = 0",
                "ABS(HorizontalDisplacement) = 0",
                "ABS(Rotation) = 0", exclusionStatement);
        List<GroundTruth> unAugmententedSamples = DbUtils.readGroundTruths(filterClause);
        // Group according to files
        Map<String, List<GroundTruth>> fileGTMap = new HashMap<>();
        for(GroundTruth gt : unAugmententedSamples)
        {
            if(!fileGTMap.containsKey(gt.fileName))
                fileGTMap.put(gt.fileName, new ArrayList<>());
            fileGTMap.get(gt.fileName).add(gt);
        }
        Mat topLeftBBCoords = new Mat(0, 2, CvType.CV_32F);
        List<List<GroundTruth>> horizontalNeighbors = new ArrayList<>();
        List<List<GroundTruth>> verticalNeighbors = new ArrayList<>();
        // Detect the top left character's bounding box for every training file.
        for(String fileName : fileGTMap.keySet())
        {
            List<GroundTruth> gtListForFile = fileGTMap.get(fileName);
            assert gtListForFile.size() % TOTAL_CHAR_COUNT == 0;
            // First sort vertically, split upper and lower rows.
            gtListForFile.sort(Comparator.comparingInt(gt0 -> gt0.getBoundingRect().y));
            // The upper half of the list belongs to upper rows.
            List<GroundTruth> upperRow  = gtListForFile.subList(0, gtListForFile.size()/2);
            List<GroundTruth> lowerRow  = gtListForFile.subList(gtListForFile.size()/2, gtListForFile.size());
            // Sort the upper row horizontally
            upperRow.sort(Comparator.comparingInt(gt -> gt.getBoundingRect().x));
            // Sort the lower row horizontally
            lowerRow.sort(Comparator.comparingInt(gt -> gt.getBoundingRect().x));
            // Handle multiple file case
            int duplicateCount = gtListForFile.size() / TOTAL_CHAR_COUNT;
            Set<List<GroundTruth>> rows = new HashSet<>();
            rows.add(upperRow);
            rows.add(lowerRow);
            // Create horizontal adjacent left right pairs
            for(List<GroundTruth> row : rows)
            {
                for(int i=0;i<TOTAL_CHAR_COUNT/2-1;i++)
                {
                    Set<GroundTruth> leftGroup = new HashSet<>(row.subList(i*duplicateCount, (i+1)*duplicateCount));
                    Set<GroundTruth> rightGroup = new HashSet<>(row.subList((i+1)*duplicateCount, (i+2)*duplicateCount));
                    Set<List<GroundTruth>> cartesianProduct =  Sets.cartesianProduct(leftGroup, rightGroup);
                    horizontalNeighbors.addAll(cartesianProduct);
                }
            }
            // Create vertical adjacent upper lower pairs
            for(int i=0;i<TOTAL_CHAR_COUNT/2;i++)
            {
                Set<GroundTruth> upperGroup = new HashSet<>(upperRow.subList(i*duplicateCount, (i+1)*duplicateCount));
                Set<GroundTruth> lowerGroup = new HashSet<>(lowerRow.subList(i*duplicateCount, (i+1)*duplicateCount));
                Set<List<GroundTruth>> cartesianProduct =  Sets.cartesianProduct(upperGroup, lowerGroup);
                verticalNeighbors.addAll(cartesianProduct);
            }
            // difs.stream().sorted(Comparator.comparingInt());
            // Get the top left bounding box. If there is multiple annotations for the same file, there can be
            // multiple coords.
            Mat upperLeftCoord = new Mat(1, 2, CvType.CV_32F);
            upperLeftCoord.put(0,0, upperRow.get(0).getBoundingRect().x);
            upperLeftCoord.put(0,1, upperRow.get(0).getBoundingRect().y);
            topLeftBBCoords.push_back(upperLeftCoord);
        }
        // Learn a Gaussian Mixture Model to model the top left bounding box' coordinate distribution.
        TermCriteria terminationCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                1000, 1e-4 );
        EM em = EM.create();//new EM(topLeftMixtureCount, EM.COV_MAT_DIAGONAL, terminationCriteria);
        em.setTermCriteria(terminationCriteria);
        em.setClustersNumber(topLeftMixtureCount);
        System.out.println(topLeftBBCoords.dump());
        TrainData trainData = TrainData.create(topLeftBBCoords, Ml.ROW_SAMPLE, new Mat());
        em.train(trainData, EM.COV_MAT_DIAGONAL);
        // Fill in the mixture data
        Mat weights = em.getWeights();
        Mat means = em.getMeans();
        List<Mat> covMatrices = new ArrayList<>();
        em.getCovs(covMatrices);
        gaussianPriors = new double[topLeftMixtureCount];
        topLeftGaussianComponents = new MultivariateNormalDistribution[topLeftMixtureCount];
        for(int i=0;i<topLeftMixtureCount;i++)
        {
            gaussianPriors[i] = weights.get(0, i)[0];
            double [] mean = {means.get(i,0)[0], means.get(i,1)[0]};
            double [][] covMatrix = {
                    {covMatrices.get(i).get(0,0)[0], covMatrices.get(i).get(0,1)[0]},
                    {covMatrices.get(i).get(1,0)[0], covMatrices.get(i).get(1,1)[0]}};
            topLeftGaussianComponents[i] = new MultivariateNormalDistribution(mean, covMatrix);
        }
        // Learn horizontal and vertical steps between bounding boxes as linear regressions.
        Mat[] designAndTargetHorizontalX = getDesignAndTargetMatrices(horizontalNeighbors, true);
        Mat[] designAndTargetHorizontalY = getDesignAndTargetMatrices(horizontalNeighbors, false);
        Mat[] designAndTargetVerticalX   = getDesignAndTargetMatrices(verticalNeighbors, true);
        Mat[] designAndTargetVerticalY   = getDesignAndTargetMatrices(verticalNeighbors, false);
        OLSResult horizontalXRes =
                fitLinearRegression(designAndTargetHorizontalX[0], designAndTargetHorizontalX[1]);
        //System.out.println(horizontalXRes.getWeights().dump());
        OLSResult horizontalYRes =
                fitLinearRegression(designAndTargetHorizontalY[0], designAndTargetHorizontalY[1]);
        //System.out.println(horizontalYRes.getWeights().dump());
        OLSResult verticalXRes =
                fitLinearRegression(designAndTargetVerticalX[0], designAndTargetVerticalX[1]);
        //System.out.println(verticalXRes.getWeights().dump());
        OLSResult verticalYRes =
                fitLinearRegression(designAndTargetVerticalY[0], designAndTargetVerticalY[1]);
        //System.out.println(verticalYRes.getWeights().dump());
        // Build the covariance matrices of the horizontal and vertical Gaussians.
        // Horizontal Gaussian
        inverseHorizontalGaussianCov = new Mat();
        horizontalGaussianCov = new Mat(2,2,CvType.CV_64F);
        horizontalGaussianCov.put(0,0,horizontalXRes.getVariance());
        horizontalGaussianCov.put(0,1,0.0);
        horizontalGaussianCov.put(1,0,0.0);
        horizontalGaussianCov.put(1,1,horizontalYRes.getVariance());
        Core.invert(horizontalGaussianCov, inverseHorizontalGaussianCov);
        // Vertical Gaussian
        inverseVerticalGaussianCov = new Mat();
        verticalGaussianCov = new Mat(2,2,CvType.CV_64F);
        verticalGaussianCov.put(0,0,verticalXRes.getVariance());
        verticalGaussianCov.put(0,1,0.0);
        verticalGaussianCov.put(1,0,0.0);
        verticalGaussianCov.put(1,1,verticalYRes.getVariance());
        Core.invert(verticalGaussianCov, inverseVerticalGaussianCov);

        //Test Mahalanobis
//        double probabilityThreshold = 0.99;
//        double thresholdMahalanobisDistance = Math.sqrt(-2.0*Math.log(1.0-probabilityThreshold));
//        double currThreshold = 1.0;
//        while(currThreshold > 0)
//        {
//            double r = Math.sqrt(-2.0*Math.log(1.0-currThreshold));
//            System.out.println("p="+currThreshold+" r="+r);
//            currThreshold -= 0.05;
//        }
//        int rows = 250;
//        int cols = 250;
//        Mat dbgImage = new Mat(rows, cols, CvType.CV_8UC3);
//        Mat mean = new Mat(2,1,CvType.CV_64F);
//        mean.put(0,0,rows/2.0);
//        mean.put(1,0,cols/2.0);
//        Mat identity = Mat.eye(2,2,CvType.CV_64F);
//        for(int i=0;i<rows;i++)
//        {
//            for(int j=0;j<cols;j++)
//            {
//                Mat p = new Mat(2,1,CvType.CV_64F);
//                p.put(0,0,(double)j);
//                p.put(1,0,(double)i);
//                double mahDist = Utils.mahalanobisDistance(p, mean, identity);
//                if(mahDist <= Constants.FILTER_ACCEPTANCE_RADIUS)
//                    dbgImage.put(i,j,255,255,255);
//                else
//                    dbgImage.put(i,j,0,0,0);
//            }
//        }
//        Utils.showImageInPopup(Utils.matToBufferedImage(dbgImage,null));
        // Write everything into the DB.
        System.out.println("XXX");
    }

    public void filter(List<Detection> detectionList)
    {
        // Sort all detections according to the top left box likelihood
        List<Pair<Detection, Double>> topLeftCandidates = new ArrayList<>();
        for(Detection detection : detectionList)
        {
            double topLeftLikelihood = getTopLeftDensity(detection.getRect().x, detection.getRect().y);
            topLeftCandidates.add(new Pair<>(detection, topLeftLikelihood));
        }
        topLeftCandidates.sort(Comparator.comparingDouble(Pair::getSecond));
        topLeftCandidates = Lists.reverse(topLeftCandidates);
        double highestTopLeftProb = topLeftCandidates.get(0).getSecond();
        topLeftCandidates = topLeftCandidates.stream().
                filter(x -> x.getSecond() >= highestTopLeftProb*Constants.TOP_LEFT_ACCEPTANCE_THRESHOLD).
                collect(Collectors.toList());
        System.out.println(topLeftCandidates.toString());
        for(Pair topLeftCandidateProbPair : topLeftCandidates)
        {
            Detection topLeftCandidate = (Detection)topLeftCandidateProbPair.getFirst();
            double probability = (double)topLeftCandidateProbPair.getSecond();
            Set<Detection> candidateSet = new HashSet<>(detectionList);
            candidateSet.remove(topLeftCandidate);
            Detection currPivot = topLeftCandidate;
            for(int i=0;i<Constants.TOTAL_CHAR_COUNT/2;i++)
            {
                // Infer the next horizontal detection.
                if(i < Constants.TOTAL_CHAR_COUNT/2-1)
                {
                    // for()
                }

            }


//            for(Detection detection : detectionList)
//            {
//                if(detection == topLeftCandidate)
//                    continue;
//
//            }
        }


    }

    private Detection getNextBestHorizontalDetection(Detection currDetection, Set<Detection> candidateDetections)
    {
        double largestProb = -1.0;
        for(Detection candidateDetection : candidateDetections)
        {


        }
    }

    private Mat[] getDesignAndTargetMatrices(List<List<GroundTruth>> pairs, boolean regressXcoord)
    {
        int sampleCount = pairs.size();
        Mat designMatrix = new Mat(sampleCount, 3, CvType.CV_64F);
        Mat targetMatrix = new Mat(sampleCount, 1, CvType.CV_64F);
        for(int i=0;i<pairs.size();i++)
        {
            double x = pairs.get(i).get(0).x;
            double y = pairs.get(i).get(0).y;
            double t = (regressXcoord) ? pairs.get(i).get(1).x : pairs.get(i).get(1).y;
            designMatrix.put(i, 0, x);
            designMatrix.put(i, 1, y);
            designMatrix.put(i, 2,1.0);
            targetMatrix.put(i, 0, t);
        }
        return new Mat[]{designMatrix, targetMatrix};
    }

    private OLSResult fitLinearRegression(Mat designMatrix, Mat target)
    {
        // Weights with Ordinary Least Squares (MLE)
        Mat designMatrixT = new Mat();
        Core.transpose(designMatrix, designMatrixT);
        Mat iM = new Mat();
        // Core.multiply(designMatrixT, designMatrix, iM);
        Core.gemm(designMatrixT, designMatrix, 1.0, new Mat(), 0.0, iM);
        Mat inverse_iM = new Mat();
        Core.invert(iM, inverse_iM);
        Mat iM2 = new Mat();
        //Core.multiply(inverse_iM, designMatrixT, iM2);
        Core.gemm(inverse_iM, designMatrixT, 1.0, new Mat(), 0.0, iM2);
        Mat weights = new Mat();
        //Core.multiply(iM2, target, weights);
        Core.gemm(iM2, target, 1.0, new Mat(), 0.0, weights);
        Mat weightsT = new Mat();
        Core.transpose(weights, weightsT);
        // Variance (MLE)
        double sum = 0.0;
        for(int i=0;i<designMatrix.rows();i++)
        {
            Mat x = designMatrix.row(i);
            double d = x.dot(weights);
            double t = target.get(i,0)[0];
            sum += Math.pow(t - d, 2.0);
        }
        double variance = sum / (double)designMatrix.rows();
        System.out.println(weights.dump());
        return new OLSResult(weights, variance, sum);
    }

    private double getTopLeftDensity(double x, double y)
    {
        double [] coord = {x,y};
        double density = 0.0;
        for(int i=0;i<this.topLeftMixtureCount;i++)
            density += gaussianPriors[i]* topLeftGaussianComponents[i].density(coord);
        return density;
    }
}
