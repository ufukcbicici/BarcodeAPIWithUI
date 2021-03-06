package bandrol_training.model;

import bandrol_training.Constants;
import bandrol_training.model.Ensembles.SVMEnsemble;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Collectors;

import static bandrol_training.Constants.DETECTIONPATH;
import static bandrol_training.Constants.OBJECT_DETECTOR_FOLDER_PATH;

public class ObjectDetector {
    // private static double negativeMaxIoU = 0.8;
    private static SVM preLoadedSvm = null;
    private static double positiveRatio = 0.2;
    private static double negativeRatio = 0.2;
    private static Map<String, SVM> detectorMap;
    static {
        detectorMap = new HashMap<>();
    }

    private static List<Detection> nonMaximaSuppression(List<Detection> preListOfDetections, double nms_iou_threshold)
    {
//        List<Detection> sortedList = preListOfDetections.stream().
//                sorted(Comparator.comparing(Detection::getResponse)).collect(Collectors.toList());
        List<Detection> sortedList = preListOfDetections.stream().
                sorted((d0,d1) -> -d0.getResponse().compareTo(d1.getResponse())).collect(Collectors.toList());
        List<Detection> maxima = new ArrayList<>();
        while (sortedList.size() > 0)
        {
            Detection mostConfidentDetection = sortedList.get(0);
            List<Detection> survivedList = new ArrayList<>();
            maxima.add(mostConfidentDetection);
            for(int i=1;i<sortedList.size();i++)
            {
                Detection candidate = sortedList.get(i);
                double iou = Utils.calculateIoU(mostConfidentDetection.getRect(), candidate.getRect());
                if(iou < nms_iou_threshold)
                    survivedList.add(candidate);
            }
            sortedList = survivedList;
        }
        return maxima;
    }

    private static Table<Integer, Integer, Mat> extractFeatures(Mat img,
                                                                int sliding_window_width,
                                                                int sliding_window_height,
                                                                double sourceImgWidth)
    {
        Table<Integer, Integer, Mat> hogTable = HashBasedTable.create();
        double upperRatio = sourceImgWidth * Constants.QR_RATIO;
        for(int i=0;i<img.rows();i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (i < upperRatio || i + sliding_window_height - 1 >= img.rows())
                    continue;
                if (j + sliding_window_width - 1 >= img.cols())
                    continue;
                Mat imgRect = img.submat(i, i + sliding_window_height, j, j + sliding_window_width);
                Mat hogFeature = HOGExtractor.extractOpenCVHogFeature(imgRect, sliding_window_width, sliding_window_height);
                hogTable.put(i,j,hogFeature);
            }
        }
        return hogTable;
    }

    public static List<Detection> detectObjects(Mat img,
                                                int sliding_window_width,
                                                int sliding_window_height,
                                                double nms_iou_threshold,
                                                double object_sign,
                                                double source_width)
    {
        SVM svm = null;
        if(preLoadedSvm==null)
            svm = SVM.load(OBJECT_DETECTOR_FOLDER_PATH+"ObjectDetector");

        Mat resultImg = img.clone();
        Table<Integer, Integer, Mat> featureTable = extractFeatures(
                img,
                sliding_window_width,
                sliding_window_height,
                source_width);
        List<Detection> listOfDetections = new ArrayList<>();
        for(Table.Cell c : featureTable.cellSet())
        {
            Mat hogFeatureT = new Mat();
            Core.transpose((Mat) c.getValue(), hogFeatureT);
            Mat hog32f = new Mat();
            hogFeatureT.convertTo(hog32f, CvType.CV_32F);
            Mat response = new Mat();
            svm.predict(hog32f, response, StatModel.RAW_OUTPUT);
            double signedDistance = response.get(0, 0)[0];
            if (object_sign * signedDistance > 0) {
                Detection detection = new Detection(
                        new Rect((int) c.getColumnKey(), (int) c.getRowKey(), sliding_window_width, sliding_window_height),
                        -signedDistance);
                listOfDetections.add(detection);
            }
        }
        List<Detection> maxima = nonMaximaSuppression(listOfDetections, nms_iou_threshold);
        for (Detection dtc : maxima) {
            Rect r = dtc.getRect();
            Imgproc.rectangle(resultImg, new Point(r.x, r.y),
                    new Point(r.x + r.width - 1, r.y + r.height - 1),
                    new Scalar(0, 0, 255));
        }
        String fileName = Utils.getNonExistingFileName(DETECTIONPATH + "detection_result.png", ".png");
        Imgcodecs.imwrite(fileName, resultImg);
        Utils.showImageInPopup(Utils.matToBufferedImage(resultImg, null));
        return maxima;
    }

//    public void loadDetectors()
//    {
//        for(String label : Constants.LABELS)
//        {
//            String detectorPath = OBJECT_DETECTOR_FOLDER_PATH+"ObjectDetector_"+label;
//            boolean doesDetectorExist =
//                    Utils.checkFileExist(detectorPath);
//            if(!doesDetectorExist)
//                continue;
//            SVM labelDetector = SVM.load(detectorPath);
//            detectorMap.put(label, labelDetector);
//        }
//    }

    public static List<Detection> detectWithEnsembles(
                                           int ensembleCount,
                                           String label,
                                           Mat img,
                                           int sliding_window_width,
                                           int sliding_window_height,
                                           double nms_iou_threshold,
                                           double object_sign,
                                           double source_width)
    {
        SVMEnsemble svmEnsemble = new SVMEnsemble(false);
        svmEnsemble.loadEnsemble(ensembleCount, label);
        Mat resultImg = img.clone();
        Table<Integer, Integer, Mat> featureTable = extractFeatures(
                img,
                sliding_window_width,
                sliding_window_height,
                source_width);
        List<Detection> listOfDetections = new ArrayList<>();
        for(Table.Cell c : featureTable.cellSet())
        {
            Mat hogFeatureT = new Mat();
            Core.transpose((Mat) c.getValue(), hogFeatureT);
            Mat hog32f = new Mat();
            hogFeatureT.convertTo(hog32f, CvType.CV_32F);
            List<Mat> predictedLabels = svmEnsemble.predictLabels(hog32f);
            List<Mat> predictedMargins = svmEnsemble.predictMargins(hog32f);
            double totalMarginResponse = 0.0;
            double totalVote = 0.0;
            for(int svmIndex=0;svmIndex<svmEnsemble.getSvmList().size();svmIndex++)
            {
                double predictedLabel = predictedLabels.get(svmIndex).get(0,0)[0];
                totalVote += predictedLabel;
                totalMarginResponse += Math.abs(predictedMargins.get(svmIndex).get(0,0)[0])*predictedLabel;
            }
            double avgMarginResponse = totalMarginResponse / (double)svmEnsemble.getSvmList().size();
            if(totalVote > 0)
            {
                Detection detection = new Detection(
                        new Rect((int) c.getColumnKey(), (int) c.getRowKey(),
                                sliding_window_width, sliding_window_height), avgMarginResponse);
                listOfDetections.add(detection);
            }
        }
        List<Detection> maxima = nonMaximaSuppression(listOfDetections, nms_iou_threshold);
        for (Detection dtc : maxima) {
            Rect r = dtc.getRect();
            Imgproc.rectangle(resultImg, new Point(r.x, r.y),
                    new Point(r.x + r.width - 1, r.y + r.height - 1),
                    new Scalar(0, 0, 255));
            int font = Core.FONT_HERSHEY_COMPLEX;
            DecimalFormat df2 = new DecimalFormat(".##");
            String doubleFormatted = df2.format(dtc.getResponse());
            Imgproc.putText(resultImg, doubleFormatted, new Point(r.x, r.y), font,
                    0.4,new Scalar(0,255,0),1);
        }
        String fileName = Utils.getNonExistingFileName(DETECTIONPATH + "detection_result.png", ".png");
        Imgcodecs.imwrite(fileName, resultImg);
        Utils.showImageInPopup(Utils.matToBufferedImage(resultImg, null));
        return maxima;
    }

    public static void trainEnsemble(int ensembleCount,
                                     double negativeMaxIoU,
                                     double sourceImgWidth,
                                     String character)
    {
        String charToTrain = "\"" + character + "\"";
        double upperRatio = sourceImgWidth * Constants.QR_RATIO;
        String exlusionStatement = "FileName NOT IN" + Utils.getFileSelectionClause();
        String positiveFilterClause = Utils.getFilterClause(
                "Label = "+charToTrain,
                    "ABS(VerticalDisplacement) < 3",
                    "ABS(HorizontalDisplacement) < 3",
                exlusionStatement);
        String negativeFilterClause = Utils.getFilterClause(
                "Label != "+charToTrain,
                "IoUWithClosestGT < " +negativeMaxIoU,
                "XCoord < " +upperRatio,
                exlusionStatement);
        List<GroundTruth> positiveSamples = DbUtils.readGroundTruths(positiveFilterClause);
        List<GroundTruth> negativeSamples = DbUtils.readGroundTruths(negativeFilterClause);
        SVMEnsemble svmEnsemble = new SVMEnsemble();
        for(int i=0;i<ensembleCount;i++)
        {
            System.out.println("Training Object Detector SVM "+i);
            Collections.shuffle(positiveSamples);
            Collections.shuffle(negativeSamples);
            int positiveSampleCount = (int)Math.round((double)positiveSamples.size() * positiveRatio);
            int negativeSampleCount = (int)Math.round((double)negativeSamples.size() * negativeRatio);
            List<GroundTruth> positiveSubset = positiveSamples.subList(0, positiveSampleCount);
            List<GroundTruth> negativeSubset = negativeSamples.subList(0, negativeSampleCount);
            System.out.println("positiveSubset Size:"+positiveSubset.size());
            System.out.println("negativeSubset Size:"+negativeSubset.size());
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
            Mat positiveFeaturesMatrix = Utils.getFeatureMatrixFromGroundTruths(positiveSubset);
            Mat negativeFeaturesMatrix = Utils.getFeatureMatrixFromGroundTruths(negativeSubset);
            System.out.println("positiveSubset Size:"+positiveFeaturesMatrix.rows());
            System.out.println("negativeSubset Size:"+negativeFeaturesMatrix.rows());
            Mat completeFeatureMatrix = new Mat();
            Core.vconcat(Arrays.asList(positiveFeaturesMatrix, negativeFeaturesMatrix), completeFeatureMatrix);
            Mat completeFeatureMatrixFloat = new Mat();
            completeFeatureMatrix.convertTo(completeFeatureMatrixFloat, CvType.CV_32F);
            Mat positiveLabelsMatrix = new Mat(positiveSubset.size(), 1, CvType. CV_32SC1);
            Mat negativeLabelsMatrix = new Mat(negativeSubset.size(), 1, CvType. CV_32SC1);
            positiveLabelsMatrix.setTo(new Scalar(1));
            negativeLabelsMatrix.setTo(new Scalar(-1));
            assert positiveFeaturesMatrix.rows() == positiveLabelsMatrix.rows();
            assert negativeFeaturesMatrix.rows() == negativeLabelsMatrix.rows();
            Mat completeLabelsMatrix = new Mat();
            Core.vconcat(Arrays.asList(positiveLabelsMatrix, negativeLabelsMatrix), completeLabelsMatrix);
            svm.trainAuto(completeFeatureMatrixFloat, Ml.ROW_SAMPLE, completeLabelsMatrix, 10,
                    C_grid, gamma_grid, p_grid, nu_grid,
                    coeff_grid,degree_grid,false);
            svmEnsemble.addModel(svm);
            System.out.println("Finished training Object Detector SVM "+i);
        }
        svmEnsemble.saveEnsemble(charToTrain);
        System.out.println("Finished training the ensemble");
    }

    public static void train(double negativeMaxIoU, double sourceImgWidth)
    {
        double upperRatio = sourceImgWidth * Constants.QR_RATIO;
        String exlusionStatement = "FileName NOT IN" + Utils.getFileSelectionClause();
        String positiveFilterClause = Utils.getFilterClause("Label != -1", exlusionStatement);
        String negativeFilterClause = Utils.getFilterClause(
                "Label = -1",
                "IoUWithClosestGT < " +negativeMaxIoU,
                "XCoord < " +upperRatio,
                exlusionStatement);
        List<GroundTruth> positiveSamples = DbUtils.readGroundTruths(positiveFilterClause);
        List<GroundTruth> negativeSamples = DbUtils.readGroundTruths(negativeFilterClause);
//        List<GroundTruth> negativeSamples = DbUtils.readGroundTruths(
//                "Label != 0 AND IoUWithClosestGT < "+negativeMaxIoU);
//        List<GroundTruth> negativeSamples = DbUtils.readGroundTruths(
//                "(Label = -1 AND IoUWithClosestGT < "+negativeMaxIoU+") OR (Label NOT IN (0,-1))" );
        //List<GroundTruth> negativeSamples = DbUtils.readGroundTruths("Label == -1 AND IoUWithClosestGT < "+negativeMaxIoU);
        System.out.println(positiveSamples.size());
        System.out.println(negativeSamples.size());

        // Sample
        int positiveSampleCount = (int)Math.round((double)positiveSamples.size() * positiveRatio);
        int negativeSampleCount = (int)Math.round((double)negativeSamples.size() * negativeRatio);
        System.out.println(positiveSampleCount);
        System.out.println(negativeSampleCount);
        Collections.shuffle(positiveSamples);
        Collections.shuffle(negativeSamples);
        List<GroundTruth> positiveSubset = positiveSamples.subList(0, positiveSampleCount);
        List<GroundTruth> negativeSubset = negativeSamples.subList(0, negativeSampleCount);
        System.out.println(positiveSubset.size());
        System.out.println(negativeSubset.size());

        SVM svm = SVM.create();
        TermCriteria terminationCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                1000, 1e-3 );
        svm.setKernel(SVM.LINEAR);

        // Prepare the training set
        int totalSampleCount = positiveSubset.size() + negativeSubset.size();
        int featureDim = negativeSubset.get(0).getHogFeature().rows();
        Mat trainingSamples = new Mat(totalSampleCount, featureDim, CvType.CV_64F);
        Mat labelMat = new Mat(totalSampleCount, 1, CvType. CV_32SC1);
        int [] labelArr = new int[totalSampleCount];
        List<Mat> transposeFeatures = new ArrayList<>();
        for(int i=0;i<positiveSubset.size();i++)
        {
            labelArr[i] = 1;
            Mat hogFeature = positiveSubset.get(i).getHogFeature();
            Mat hogFeatureT = new Mat();
            Core.transpose(hogFeature, hogFeatureT);
            transposeFeatures.add(hogFeatureT);
        }
        for(int i=0;i<negativeSubset.size();i++)
        {
            labelArr[positiveSubset.size()+i] = -1;
            Mat hogFeature = negativeSubset.get(i).getHogFeature();
            Mat hogFeatureT = new Mat();
            Core.transpose(hogFeature, hogFeatureT);
            transposeFeatures.add(hogFeatureT);
        }
        Core.vconcat(transposeFeatures, trainingSamples);
        labelMat.put(0,0,labelArr);
        Mat trainingSamplesF = new Mat();
        trainingSamples.convertTo(trainingSamplesF, CvType.CV_32F);

        // Train the svm
        // public  boolean trainAuto(Mat samples, int layout, Mat responses, int kFold, ParamGrid Cgrid,
        // ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid, ParamGrid degreeGrid, boolean balanced)
        ParamGrid C_grid = SVM.getDefaultGridPtr(SVM.C);
        double log_step = C_grid.get_logStep();
        double min_val = C_grid.get_minVal();
        double max_val = C_grid.get_maxVal();
        ParamGrid gamma_grid = ParamGrid.create(0, 0,0);
        ParamGrid p_grid = ParamGrid.create(0, 0,0);
        ParamGrid nu_grid = ParamGrid.create(0, 0,0);
        ParamGrid coeff_grid = ParamGrid.create(0, 0,0);
        ParamGrid degree_grid = ParamGrid.create(0, 0,0);
        svm.trainAuto(trainingSamplesF, Ml.ROW_SAMPLE, labelMat, 10,
                C_grid, gamma_grid, p_grid, nu_grid,
                coeff_grid,degree_grid,false);
        System.out.println("Training completed.");
        svm.save(OBJECT_DETECTOR_FOLDER_PATH+"ObjectDetector");
        preLoadedSvm = svm;
        // Utils.getNonExistingFileName(OBJECT_DETECTOR_FOLDER_PATH, )
    }
}
