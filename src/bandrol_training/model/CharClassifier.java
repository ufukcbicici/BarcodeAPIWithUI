package bandrol_training.model;

import bandrol_training.Constants;
import org.opencv.core.*;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.util.*;
import java.util.stream.Collectors;

//Apply bagging with SVM ensembles.

public class CharClassifier
{
    public SVMEnsemble train(int ensembleCount, double sampleRatio, double minNumOfSamplesPerClass, double validationRatio)
    {
        List<GroundTruth> allSamples = DbUtils.readGroundTruths("Label != -1");
        Collections.shuffle(allSamples);
        int trainingSetSize = (int)(allSamples.size() * validationRatio);
        List<GroundTruth> trainingSet = allSamples.subList(0,trainingSetSize);
        List<GroundTruth> validationSet = allSamples.subList(trainingSetSize, allSamples.size());

        Map<String, List<GroundTruth>> samplesPerClass = new HashMap<>();
        for(GroundTruth gt : trainingSet)
        {
            if(!samplesPerClass.containsKey(gt.label))
                samplesPerClass.put(gt.label, new ArrayList<>());
            samplesPerClass.get(gt.label).add(gt);
        }
        // Apply bagging
        SVMEnsemble svmEnsemble = new SVMEnsemble();
        for(int currEnsembleIndex = 0; currEnsembleIndex < ensembleCount; currEnsembleIndex++)
        {
            List<Mat> sampleMatrices = new ArrayList<>();
            List<Mat> labels = new ArrayList<>();
            // Pick samples from each class
            for(String character : Constants.LABELS)
            {
                List<GroundTruth> classSamples = samplesPerClass.get(character);
                int classSampleCount = classSamples.size();
                int subsampleCount = (int)Math.min(Math.max(sampleRatio * (double)classSampleCount, minNumOfSamplesPerClass),classSampleCount);
                Collections.shuffle(classSamples);
                List<GroundTruth> classSubset = classSamples.subList(0, subsampleCount);
                Mat classFeaturesCombined = Utils.getFeatureMatrixFromGroundTruths(classSubset);
                Mat labelMat = new Mat(subsampleCount, 1, CvType. CV_32SC1);
                labelMat.setTo(new Scalar(Constants.CHAR_TO_LABEL_MAP.get(character)));
                System.out.println(labelMat.dump());
                sampleMatrices.add(classFeaturesCombined);
                labels.add(labelMat);
            }
            Mat totalSampleMatrix = new Mat();
            Mat totalLabelMatrix = new Mat();
            Core.vconcat(sampleMatrices, totalSampleMatrix);
            Core.vconcat(labels, totalLabelMatrix);
            // Train SVM
            ParamGrid C_grid = SVM.getDefaultGridPtr(SVM.C);
            ParamGrid gamma_grid = ParamGrid.create(0, 0,0);
            ParamGrid p_grid = ParamGrid.create(0, 0,0);
            ParamGrid nu_grid = ParamGrid.create(0, 0,0);
            ParamGrid coeff_grid = ParamGrid.create(0, 0,0);
            ParamGrid degree_grid = ParamGrid.create(0, 0,0);
            SVM svm = SVM.create();
//            TermCriteria terminationCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
//                    1000, 1e-3 );
            svm.setKernel(SVM.LINEAR);
            System.out.println("Training "+currEnsembleIndex+". SVM of the ensemble.");
            Mat totalSampleMatrixFloat = new Mat();
            totalSampleMatrix.convertTo(totalSampleMatrixFloat, CvType.CV_32F);
            svm.trainAuto(totalSampleMatrixFloat, Ml.ROW_SAMPLE, totalLabelMatrix, 10,
                    C_grid, gamma_grid, p_grid, nu_grid,
                    coeff_grid,degree_grid,false);
            System.out.println("Training of the SVM finished.");
            svm.save(Constants.CLASSIFIER_SVM_PATH + "svm_"+currEnsembleIndex);
            svmEnsemble.add(svm);
        }
        return svmEnsemble;
    }

    public void predict(SVMEnsemble ensemble, List<GroundTruth> predictionList)
    {
        Mat featureMatrix = Utils.getFeatureMatrixFromGroundTruths(predictionList);
        Mat featureMatrixFloat = new Mat();
        featureMatrix.convertTo(featureMatrixFloat, CvType.CV_32F);
        Mat responses = ensemble.predictByVoting(featureMatrixFloat);
        int totalCorrectCount = 0;
        for(int i=0;i<predictionList.size();i++)
        {
            int trueLabel = Constants.CHAR_TO_LABEL_MAP.get(predictionList.get(i).label);
            int inferredLabel = (int)responses.get(i,0)[0];
            if(inferredLabel == trueLabel)
                totalCorrectCount++;
        }
        System.out.println("Total Samples:"+predictionList.size());
        System.out.println("Total Correct Count:"+totalCorrectCount);
        System.out.println("Accuracy:"+(double)totalCorrectCount / (double)predictionList.size());
    }
}
