package bandrol_training.post_processing;

import bandrol_training.model.DbUtils;
import bandrol_training.model.GroundTruth;
import bandrol_training.model.Utils;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.EM;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static bandrol_training.Constants.TOTAL_CHAR_COUNT;

public class ProbabilisticPostProcesser {

    private int topLeftMixtureCount;
    private double [] gaussianPriors;
    private MultivariateNormalDistribution [] gaussians;

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
        gaussians = new MultivariateNormalDistribution[topLeftMixtureCount];
        for(int i=0;i<topLeftMixtureCount;i++)
        {
            gaussianPriors[i] = weights.get(0, i)[0];
            double [] mean = {means.get(i,0)[0], means.get(i,1)[0]};
            double [][] covMatrix = {
                    {covMatrices.get(i).get(0,0)[0], covMatrices.get(i).get(0,1)[0]},
                    {covMatrices.get(i).get(1,0)[0], covMatrices.get(i).get(1,1)[0]}};
            gaussians[i] = new MultivariateNormalDistribution(mean, covMatrix);
        }
        // Learn horizontal and vertical steps between bounding boxes as linear regressions.
        // Step 1): Horizontal


        System.out.println("XXX");
    }

    private double fitLinearRegression(Mat sample, Mat target, Mat weights)
    {
        // Weights with Ordinary Least Squares (MLE)
        Mat designMatrix = new Mat(sample.rows(), sample.cols()+1, sample.type());
        for(int i=0;i<sample.rows();i++)
        {
            for(int j=0;j<sample.cols();j++)
                designMatrix.put(i,j,sample.get(i,j)[0]);
            designMatrix.put(i,sample.cols(),1.0);
        }
        Mat designMatrixT = new Mat();
        Core.transpose(designMatrix, designMatrixT);
        Mat iM = new Mat();
        Core.multiply(designMatrixT, designMatrix, iM);
        Mat inverse_iM = new Mat();
        Core.invert(iM, inverse_iM);
        Mat iM2 = new Mat();
        Core.multiply(inverse_iM, designMatrixT, iM2);
        weights = new Mat();
        Core.multiply(iM2, target, weights);
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
        double variance = (double)designMatrix.rows() / sum;
        return variance;
    }

    private double getTopLeftDensity(double x, double y)
    {
        double [] coord = {x,y};
        double density = 0.0;
        for(int i=0;i<this.topLeftMixtureCount;i++)
            density += gaussianPriors[i]*gaussians[i].density(coord);
        return density;
    }
}
