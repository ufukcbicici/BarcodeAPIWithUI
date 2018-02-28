package bandrol_training.model.Ensembles;

import bandrol_training.Constants;
import org.apache.commons.math3.random.EmpiricalDistribution;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SVMEnsemble extends EnsembleModel
{
    private boolean isMultiClass;
    // private static final double BIN_WIDTH = 0.05;
    private Map<SVM, EmpiricalDistribution> distributionMap;

    public SVMEnsemble(boolean isMultiClass)
    {
        this.models = new ArrayList<>();
        this.isMultiClass = isMultiClass;
        distributionMap = new HashMap<>();
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
            Mat responses = new Mat();
            svm.predict(samples, responses, 0);
            for(int i=0;i<responses.rows();i++)
            {
                int responseLabel = (int)responses.get(i,0)[0];
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

    public void addModel(StatModel model, Mat trainingSamples) throws Exception
    {
        // Build empirical response distribution with "samples", if this is a binary SVM ensemble.
        if(isMultiClass)
            throw new Exception("Weighted Voting is undefined for multiclass SVM Ensembles.");
        addModel(model);
        Mat X = makeSamplesCompatible(trainingSamples);
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
        Mat marginMatrix = predictMargins(samplesT32f);
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

    public void saveEnsemble(String token)
    {
        for(int i=0;i<this.models.size();i++)
        {
            this.models.get(i).save(Constants.OBJECT_DETECTOR_FOLDER_PATH + "object_detector_for_" + token +"_svm_"+i);
        }
    }

    private Mat predictLabels(Mat samples)
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

    private Mat predictMargins(Mat samples)
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
        Mat samplesT = new Mat();
        Core.transpose(samples, samplesT);
        Mat samplesT32f = new Mat();
        samplesT.convertTo(samplesT32f, CvType.CV_32F);
        return samplesT32f;
    }
}
