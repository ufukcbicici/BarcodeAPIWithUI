package bandrol_training.model;

import bandrol_training.Constants;
import bandrol_training.model.Ensembles.SVMEnsemble;
import org.opencv.core.*;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;

import java.util.*;
import java.util.stream.Collectors;

//Apply bagging with SVM ensembles.

public class CharClassifier
{
    public static SVMEnsemble train(int ensembleCount, double sampleRatio, double minNumOfSamplesPerClass,
                                    double validationRatio)
    {
        SVMEnsemble svmEnsemble = new SVMEnsemble(true);
        String inclusionStatement = "FileName IN" + Utils.getFileSelectionClause();
        String exclusionStatement = "FileName NOT IN" + Utils.getFileSelectionClause();
        String includeClause = Utils.getFilterClause(
                "Label != -1",
                "ABS(VerticalDisplacement) <= 1",
                "ABS(HorizontalDisplacement) <= 1",
                inclusionStatement);
        String excludeClause = Utils.getFilterClause(
                "Label != -1",
                "ABS(VerticalDisplacement) <= 1",
                "ABS(HorizontalDisplacement) <= 1",
                exclusionStatement);
        List<GroundTruth> allTrainingSamples = DbUtils.readGroundTruths(excludeClause);
        List<GroundTruth> allTestSamples = DbUtils.readGroundTruths(includeClause);
        System.out.println("All Training Samples:" + allTrainingSamples.size());
        System.out.println("Test Set:" + allTestSamples.size());
        Collections.shuffle(allTrainingSamples);
        int trainingSetSize = (int)(allTrainingSamples.size() * (1.0-validationRatio));
        List<GroundTruth> trainingSet = allTrainingSamples.subList(0,trainingSetSize);
        List<GroundTruth> validationSet = allTrainingSamples.subList(trainingSetSize, allTrainingSamples.size());
        List<GroundTruth> unrotatedTestSamples = allTestSamples.stream().
                filter(gt -> Math.abs(gt.rotation) < 0.0001).collect(Collectors.toList());
        System.out.println("Training Set:"+trainingSet.size());
        System.out.println("Validation Set:"+validationSet.size());
        Map<String, List<GroundTruth>> samplesPerClass = new HashMap<>();
        for(GroundTruth gt : trainingSet)
        {
            if(!samplesPerClass.containsKey(gt.label))
                samplesPerClass.put(gt.label, new ArrayList<>());
            samplesPerClass.get(gt.label).add(gt);
        }
//        // Apply bagging
//        SVMEnsemble svmEnsemble = new SVMEnsemble();
//        for(int currEnsembleIndex = 0; currEnsembleIndex < ensembleCount; currEnsembleIndex++)
//        {
//            System.out.println("***********SVM "+currEnsembleIndex+"***********");
//            List<Mat> sampleMatrices = new ArrayList<>();
//            List<Mat> labels = new ArrayList<>();
//            // Pick samples from each class
//            Set<String> targetLabels = Set.of("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J",
//                    "K","M","N","P","Q","R","S","T","V","W","X","Y","Z");
//            for(String character : targetLabels)
//            {
//                if(!samplesPerClass.containsKey(character))
//                    continue;
//                List<GroundTruth> classSamples = samplesPerClass.get(character);
//                int classSampleCount = classSamples.size();
//                int subsampleCount = (int)Math.max(sampleRatio * (double)classSampleCount, minNumOfSamplesPerClass);
//                Collections.shuffle(classSamples);
//                List<GroundTruth> classSubset = classSamples.subList(0, subsampleCount);
//                Mat classFeaturesCombined = Utils.getFeatureMatrixFromGroundTruths(classSubset);
//                Mat labelMat = new Mat(subsampleCount, 1, CvType. CV_32SC1);
//                labelMat.setTo(new Scalar(Constants.CHAR_TO_LABEL_MAP.get(character)));
//                // System.out.println(labelMat.dump());
//                sampleMatrices.add(classFeaturesCombined);
//                labels.add(labelMat);
//            }
//            Mat totalSampleMatrix = new Mat();
//            Mat totalLabelMatrix = new Mat();
//            Core.vconcat(sampleMatrices, totalSampleMatrix);
//            Core.vconcat(labels, totalLabelMatrix);
//            // Train SVM
//            ParamGrid C_grid = SVM.getDefaultGridPtr(SVM.C);
//            // ParamGrid gamma_grid = ParamGrid.create(0, 0,0);
//            ParamGrid gamma_grid = SVM.getDefaultGridPtr(SVM.GAMMA);
//            ParamGrid p_grid = ParamGrid.create(0, 0,0);
//            ParamGrid nu_grid = ParamGrid.create(0, 0,0);
//            ParamGrid coeff_grid = ParamGrid.create(0, 0,0);
//            ParamGrid degree_grid = ParamGrid.create(0, 0,0);
//            SVM svm = SVM.create();
////            TermCriteria terminationCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
////                    1000, 1e-3 );
//            svm.setKernel(SVM.RBF);
//            System.out.println("Training "+currEnsembleIndex+". SVM of the ensemble.");
//            Mat totalSampleMatrixFloat = new Mat();
//            totalSampleMatrix.convertTo(totalSampleMatrixFloat, CvType.CV_32F);
//            svm.trainAuto(totalSampleMatrixFloat, Ml.ROW_SAMPLE, totalLabelMatrix, 10,
//                    C_grid, gamma_grid, p_grid, nu_grid,
//                    coeff_grid,degree_grid,false);
//            System.out.println("Training of the SVM finished.");
//            svm.save(Constants.CLASSIFIER_SVM_PATH + "svm_"+currEnsembleIndex);
//            svmEnsemble.add(svm);
//            //Measure training set performance
//            double trainingAccuracy = predict(svm, trainingSet, targetLabels);
//            System.out.println("Training Accuracy:"+trainingAccuracy);
//            //Measure validation set performance
//            if(validationSet.size() > 0)
//            {
//                double validationAccuracy = predict(svm, validationSet, targetLabels);
//                System.out.println("Validation Accuracy:"+validationAccuracy);
//            }
//            //Measure test set performance
//            double testAccuracy = predict(svm, allTestSamples, targetLabels);
//            System.out.println("Test Accuracy:"+testAccuracy);
//            //Measure unrotated set performance
//            double unrotatedTestAccuracy = predict(svm, unrotatedTestSamples, targetLabels);
//            System.out.println("Unrotated Test Accuracy:"+unrotatedTestAccuracy);
//            System.out.println("***********SVM "+currEnsembleIndex+"***********");
//        }
//        predict(svmEnsemble, trainingSet);
//        if(validationSet.size() > 0)
//            predict(svmEnsemble, validationSet);
//        predict(svmEnsemble, allTestSamples);
//        //Measure unrotated set performance
//        predict(svmEnsemble, unrotatedTestSamples);
//        return svmEnsemble;
        return  svmEnsemble;
    }

    public static double predict(SVM svm, List<GroundTruth> predictionList, Set<String> targetLabels)
    {
        List<GroundTruth> filteredList = predictionList.stream().filter(
                gt -> targetLabels.contains(gt.label)).collect(Collectors.toList());
//        Mat featureMatrix = new Mat(0, filteredList.get(0).getHogFeature().rows(), CvType.CV_64F);
//        for (int i=0;i<predictionList.size();i++)
//        {
//            GroundTruth gt = predictionList.get(i);
//            Mat hogFeatureT = new Mat();
//            Core.transpose(gt.getHogFeature(), hogFeatureT);
//            featureMatrix.push_back(hogFeatureT);
//        }
        Mat featureMatrix = Utils.getFeatureMatrixFromGroundTruths(predictionList);
        Mat featureMatrixF = new Mat();
        featureMatrix.convertTo(featureMatrixF, CvType.CV_32F);
        Mat validationResponses = new Mat();
        svm.predict(featureMatrixF, validationResponses, 0);
        int totalCorrectCount = 0;
        for(int i=0;i<filteredList.size();i++)
        {
            int trueLabel = Constants.CHAR_TO_LABEL_MAP.get(filteredList.get(i).label);
            if(trueLabel == validationResponses.get(i,0)[0])
                totalCorrectCount++;
        }
        return (double)totalCorrectCount / (double)filteredList.size();
    }

    public static void predict(SVMEnsemble ensemble, List<GroundTruth> predictionList)
    {
        Mat featureMatrix = Utils.getFeatureMatrixFromGroundTruths(predictionList);
        Mat featureMatrixFloat = new Mat();
        featureMatrix.convertTo(featureMatrixFloat, CvType.CV_32F);
//        Mat validationResponses = new Mat();
//        ensemble.getSvmList().get(0).predict(featureMatrixFloat, validationResponses, 0);
        Mat responses = ensemble.predictByVoting(featureMatrixFloat);
        int totalCorrectCount = 0;
        for(int i=0;i<predictionList.size();i++)
        {
            int trueLabel = Constants.CHAR_TO_LABEL_MAP.get(predictionList.get(i).label);
            if(trueLabel == responses.get(i,0)[0])
                totalCorrectCount++;
        }

//        Mat responses = ensemble.predictByVoting(featureMatrixFloat);
//        int totalCorrectCount = 0;
//        for(int i=0;i<predictionList.size();i++)
//        {
//            int trueLabel = Constants.CHAR_TO_LABEL_MAP.get(predictionList.get(i).label);
//            int inferredLabel = (int)responses.get(i,0)[0];
//            if(inferredLabel == trueLabel)
//                totalCorrectCount++;
//        }
        System.out.println("Total Samples:"+predictionList.size());
        System.out.println("Total Correct Count:"+totalCorrectCount);
        System.out.println("Accuracy:"+(double)totalCorrectCount / (double)predictionList.size());
    }
}
