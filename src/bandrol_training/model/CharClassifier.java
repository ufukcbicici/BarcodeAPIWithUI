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
    public static SVMEnsemble train(int ensembleCount, double sampleRatio, double minNumOfSamplesPerClass, double validationRatio)
    {
        List<GroundTruth> allSamples = DbUtils.readGroundTruths("Label != -1");
        System.out.println(allSamples.size());
        Collections.shuffle(allSamples);
        int trainingSetSize = (int)(allSamples.size() * (1.0-validationRatio));
        List<GroundTruth> trainingSet = allSamples.subList(0,trainingSetSize);
        List<GroundTruth> validationSet = allSamples.subList(trainingSetSize, allSamples.size());
        System.out.println(trainingSet.size());
        System.out.println(validationSet.size());
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
            Set<String> targetLabels = Set.of("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J",
                    "K","M","N","P","Q","R","S","T","V","W","X","Y","Z");
            for(String character : targetLabels)
            {
                if(!samplesPerClass.containsKey(character))
                    continue;
                List<GroundTruth> classSamples = samplesPerClass.get(character);
                int classSampleCount = classSamples.size();
                int subsampleCount = (int)Math.max(sampleRatio * (double)classSampleCount, minNumOfSamplesPerClass);
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

            //Measure training set performance
            Mat trainingResponses = new Mat();
            svm.predict(totalSampleMatrixFloat,trainingResponses,0);
            int trainingTotal = 0;
            int trainingTotalCorrectCount = 0;
            for(int i=0;i<totalLabelMatrix.rows();i++)
            {
                trainingTotal++;
                if(totalLabelMatrix.get(i,0)[0] == trainingResponses.get(i,0)[0])
                    trainingTotalCorrectCount++;
            }
            double trainingAccuracy = (double)trainingTotalCorrectCount / (double)trainingTotal;
            System.out.println("Training Accuracy:"+trainingAccuracy);
            //Measure validaiton set performance
            List<GroundTruth> targetLabelValidationSet = validationSet.stream().filter(gt -> targetLabels.contains(gt.label)).collect(Collectors.toList());
            Mat featureMatrix = new Mat(0, targetLabelValidationSet.get(0).getHogFeature().rows(), CvType.CV_64F);
            for(int i=0;i<targetLabelValidationSet.size();i++)
            {
                Mat hogFeatureT = new Mat();
                Core.transpose(targetLabelValidationSet.get(i).getHogFeature(), hogFeatureT);
                featureMatrix.push_back(hogFeatureT);
            }
            Mat featureMatrixF = new Mat();
            featureMatrix.convertTo(featureMatrixF, CvType.CV_32F);
            Mat validationResponses = new Mat();
            svm.predict(featureMatrixF, validationResponses, 0);
            int validationTotal = 0;
            int validationTotalCorrectCount = 0;
            for(int i=0;i<targetLabelValidationSet.size();i++)
            {
                validationTotal++;
                int trueLabel = Constants.CHAR_TO_LABEL_MAP.get(targetLabelValidationSet.get(i).label);
                if(trueLabel == validationResponses.get(i,0)[0])
                    validationTotalCorrectCount++;
            }
            double validationAccuracy = (double)validationTotalCorrectCount / (double)validationTotal;
            System.out.println("Validation Accuracy:"+validationAccuracy);

        }
        //predict(svmEnsemble, validationSet);
        return svmEnsemble;
    }

    public static void predict(SVMEnsemble ensemble, List<GroundTruth> predictionList)
    {
//        Mat featureMatrix = Utils.getFeatureMatrixFromGroundTruths(predictionList);
//        Mat featureMatrixFloat = new Mat();
//        featureMatrix.convertTo(featureMatrixFloat, CvType.CV_32F);
//
//        // Mat responses = ensemble.predictByVoting(featureMatrixFloat);
//        int totalCorrectCount = 0;
//        for(int i=0;i<predictionList.size();i++)
//        {
//            int trueLabel = Constants.CHAR_TO_LABEL_MAP.get(predictionList.get(i).label);
//            int inferredLabel = (int)responses.get(i,0)[0];
//            if(inferredLabel == trueLabel)
//                totalCorrectCount++;
//        }
//        System.out.println("Total Samples:"+predictionList.size());
//        System.out.println("Total Correct Count:"+totalCorrectCount);
//        System.out.println("Accuracy:"+(double)totalCorrectCount / (double)predictionList.size());
    }
}
