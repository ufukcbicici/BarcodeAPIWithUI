package bandrol_training.model.Ensembles;

import bandrol_training.Constants;
import bandrol_training.model.DbUtils;
import bandrol_training.model.Detectors.SVMInfo;
import bandrol_training.model.GroundTruth;
import bandrol_training.model.Utils;
import org.apache.commons.math3.random.EmpiricalDistribution;
import org.opencv.core.*;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.util.*;

public class SVMEnsemble extends EnsembleModel
{
    private boolean isMultiClass;
    // private static final double BIN_WIDTH = 0.05;
    private Map<SVM, EmpiricalDistribution> distributionMap;
    private Map<SVM, Double> positiveSignMap;

    public SVMEnsemble(boolean isMultiClass)
    {
        this.models = new ArrayList<>();
        this.isMultiClass = isMultiClass;
        distributionMap = new HashMap<>();
        positiveSignMap = new HashMap<>();
    }

    public SVMEnsemble(List<SVM> svmList, boolean isMultiClass)
    {
        this.models = new ArrayList<>();
        this.models.addAll(svmList);
        this.isMultiClass = isMultiClass;
        distributionMap = new HashMap<>();
    }

    public List<StatModel> getSvmList()
    {
        return this.models;
    }

    public Mat predictByVoting(Mat samples)
    {
        //svm.predict(hog32f, response, StatModel.RAW_OUTPUT);
        Mat voteTable = Mat.zeros(samples.rows(), Constants.LABELS.size(), CvType.CV_32S);
        for(StatModel statModel : models)
        {
            SVM svm = (SVM)statModel;
            Mat labelResponses = new Mat();
            svm.predict(samples, labelResponses, 0);
            for(int i=0;i<labelResponses.rows();i++)
            {
                int responseLabel = (int)labelResponses.get(i,0)[0];
                int labelVoteCount = (int)voteTable.get(i, responseLabel)[0];
                voteTable.put(i, responseLabel, labelVoteCount+1);
            }
        }
        Mat finalResponses = new Mat(samples.rows(), 1, CvType.CV_32S);
        for(int i=0;i<samples.rows();i++)
        {
            Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(voteTable.row(i));
            int label = (int)minMaxLocResult.maxLoc.x;
            finalResponses.put(i, 0, label);
        }
        return finalResponses;
    }

    private void addModel(StatModel model, Mat trainingSamples, boolean convertSamples) throws Exception
    {
        // Build empirical response distribution with "samples", if this is a binary SVM ensemble.
        if(isMultiClass)
            throw new Exception("Weighted Voting is undefined for multiclass SVM Ensembles.");
        addModel(model);
        Mat X = (convertSamples)?makeSamplesCompatible(trainingSamples):trainingSamples;
        Mat responses = new Mat();
        model.predict(X, responses, StatModel.RAW_OUTPUT);
        EmpiricalDistribution marginDistribution = new EmpiricalDistribution();
        double [] responseArr = new double[responses.rows()];
        for(int i=0;i<responses.rows();i++)
            responseArr[i]=responses.get(i,0)[0];
        marginDistribution.load(responseArr);
        distributionMap.put((SVM)model, marginDistribution);
    }

    public Mat predictByWeightedVoting(Mat samples) throws Exception {
        if(isMultiClass)
            throw new Exception("Weighted Voting is undefined for multiclass SVM Ensembles.");
        Mat samplesT32f = makeSamplesCompatible(samples);
        Mat labelMatrix = predictLabels(samplesT32f);
        Mat marginMatrix = predictConfidences(samplesT32f);
        Mat resultMatrix = new Mat(samplesT32f.rows(), 2, CvType.CV_64F);
        for(int i=0;i<samples.rows();i++)
        {
            int j = 0;
            // 1 or -1
            double totalLabel = 0.0;
            // Normalized response strength
            double marginalProbability = 0.0;
            for(StatModel statModel : models)
            {
                double cumulativeModelProb = marginMatrix.get(i,j)[0];
                double sampleLabel = labelMatrix.get(i,j)[0];
                marginalProbability += cumulativeModelProb;
                totalLabel += sampleLabel;
                j++;
            }
            marginalProbability /= (double)models.size();
            resultMatrix.put(i,0, totalLabel);
            resultMatrix.put(i,1,marginalProbability);
        }
        return resultMatrix;
    }

    public void loadEnsemble(int ensembleCount, String label)
    {
        for(int i=0;i<ensembleCount;i++)
        {
            SVM svm = SVM.load(Constants.OBJECT_DETECTOR_FOLDER_PATH + "object_detector_for_" + label +"_svm_"+i);
            assert svm != null;
            this.models.add(svm);
        }
    }

    // Load all SVMs it can find.
    public void loadEnsemble(String label)
    {
//        int currSVMIndex = 0;
//        while (true)
//        {
//            String svmPath = Constants.OBJECT_DETECTOR_FOLDER_PATH + "object_detector_for_" + label +"_svm_"+currSVMIndex;
//            boolean doesExist = Utils.checkFileExist(svmPath);
//            if(!doesExist)
//            {
//                break;
//            }
//            SVM svm = SVM.load(svmPath);
//            this.models.add(svm);
//            currSVMIndex++;
//        }
        List<SVMInfo> svmInfoList = DbUtils.readObjectDetectionSvms(label);
        for(SVMInfo svmInfo : svmInfoList)
        {
            String svmPath = Constants.OBJECT_DETECTOR_FOLDER_PATH + svmInfo.getFileName();
            double positiveLabelSign = svmInfo.getPositiveSign();
            SVM svm = SVM.load(svmPath);
            this.models.add(svm);
            this.positiveSignMap.put(svm, positiveLabelSign);
        }
    }

    public void saveEnsemble(String token)
    {
        int base = 0;
        for(int i=0;i<this.models.size();i++)
        {
            int id = base + i;
            String svmPath = Constants.OBJECT_DETECTOR_FOLDER_PATH + "object_detector_for_" + token +"_svm_"+id;
            while(Utils.checkFileExist(svmPath))
            {
                base++;
                id = base + i;
                svmPath = Constants.OBJECT_DETECTOR_FOLDER_PATH + "object_detector_for_" + token +"_svm_"+id;
            }
            this.models.get(i).save(svmPath);
        }
    }

    public void trainSingleModel(Mat samples, Mat labels)
    {
        SVM svm = SVM.create();
        TermCriteria terminationCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                1000, 1e-3 );
        svm.setKernel(SVM.LINEAR);
        ParamGrid C_grid = SVM.getDefaultGridPtr(SVM.C);
        ParamGrid gamma_grid = ParamGrid.create(0, 0,0);
        ParamGrid p_grid = ParamGrid.create(0, 0,0);
        ParamGrid nu_grid = ParamGrid.create(0, 0,0);
        ParamGrid coeff_grid = ParamGrid.create(0, 0,0);
        ParamGrid degree_grid = ParamGrid.create(0, 0,0);
        Mat compatibleSamples = makeSamplesCompatible(samples);
        svm.trainAuto(compatibleSamples, Ml.ROW_SAMPLE, labels, 10,
                C_grid, gamma_grid, p_grid, nu_grid,
                coeff_grid,degree_grid,false);
        addModel(svm);
//        if(this.isMultiClass)
//            addModel(svm);
//        else
//        {
//            try
//            {
//                addModel(svm, compatibleSamples, false);
//            }
//            catch (Exception e)
//            {
//                e.printStackTrace();
//            }
//        }
    }

    public Mat predictLabels(Mat samples)
    {
        List<Mat> responses = new ArrayList<>();
        for(StatModel statModel : models)
        {
            SVM svm = (SVM)statModel;
            Mat response = new Mat();
            svm.predict(samples, response, 0);
            responses.add(response);
        }
        Mat combinedResponseMatrix = new Mat();
        Core.hconcat(responses, combinedResponseMatrix);
        return combinedResponseMatrix;
    }

    public Mat predictConfidences(Mat samples)
    {
        List<Mat> margins = new ArrayList<>();
        for(StatModel statModel : models)
        {
            SVM svm = (SVM)statModel;
            Mat response = new Mat();
            svm.predict(samples, response, StatModel.RAW_OUTPUT);
            margins.add(response);
        }
        Mat combinedMarginMatrix = new Mat();
        Core.hconcat(margins, combinedMarginMatrix);
        return combinedMarginMatrix;
    }

    private Mat makeSamplesCompatible(Mat samples)
    {
//        Mat samplesT = new Mat();
//        Core.transpose(samples, samplesT);
        Mat samplesT32f = new Mat();
        samples.convertTo(samplesT32f, CvType.CV_32F);
        return samplesT32f;
    }
}
