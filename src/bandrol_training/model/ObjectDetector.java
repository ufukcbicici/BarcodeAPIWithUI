package bandrol_training.model;

import bandrol_training.Constants;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.util.*;
import java.util.stream.Collectors;

import static bandrol_training.Constants.DEBUGPATH;
import static bandrol_training.Constants.DETECTIONPATH;
import static bandrol_training.Constants.OBJECT_DETECTOR_FOLDER_PATH;

class Detection
{
    private Rect rect;
    private Double response;

    public Detection(Rect rect, double response)
    {
        this.rect = rect;
        this.response = response;
    }

    public Rect getRect() {
        return rect;
    }

    public Double getResponse() {
        return response;
    }
}

public class ObjectDetector {
    // private static double negativeMaxIoU = 0.8;
    private static SVM preLoadedSvm = null;
    private static double positiveRatio = 0.2;
    private static double negativeRatio = 0.1;
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

    public static List<Detection> detectObjects(Mat img,
                                                int sliding_window_width, int sliding_window_height,
                                                double nms_iou_threshold, double object_sign)
    {
        Mat resultImg = img.clone();
        SVM svm = null;
        if(preLoadedSvm==null)
            svm = SVM.load(OBJECT_DETECTOR_FOLDER_PATH+"ObjectDetector_0");
        List<Detection> listOfDetections = new ArrayList<>();
        for(int i=0;i<img.rows();i++)
        {
            for(int j=0;j<img.cols();j++)
            {
                if(i + sliding_window_height - 1 >= img.rows())
                    continue;
                if(j + sliding_window_width - 1 >= img.cols())
                    continue;
                Mat imgRect = img.submat(i, i + sliding_window_height, j, j + sliding_window_width);
                Mat hogFeature = HOGExtractor.extractOpenCVHogFeature(imgRect, sliding_window_width,
                        sliding_window_height);
                Mat hogFeatureT = new Mat();
                Core.transpose(hogFeature, hogFeatureT);
                Mat hog32f = new Mat();
                hogFeatureT.convertTo(hog32f, CvType.CV_32F);
                Mat response = new Mat();
                svm.predict(hog32f, response, StatModel.RAW_OUTPUT);
//                svm.predict(hog32f, response, 0);
//                if(response.get(0,0)[0] == 1)
//                {
//                    //Utils.drawLineOnMat(resultImg, up0, up1, new Scalar(255,255,0), 1);
//                    Imgproc.rectangle(resultImg, new Point(j,i),
//                            new Point(j+sliding_window_width - 1, i + (int)sliding_window_height - 1),
//                            new Scalar(255,255,0));
//                }
                double signedDistance = response.get(0,0)[0];
                if(object_sign*signedDistance > 0)
                {
                    Detection detection = new Detection(
                            new Rect(j,i,sliding_window_width,sliding_window_height), -signedDistance);
                    listOfDetections.add(detection);
                }
            }
        }
        List<Detection> maxima = nonMaximaSuppression(listOfDetections, nms_iou_threshold);
        for(Detection dtc : maxima)
        {
            Rect r = dtc.getRect();
            Imgproc.rectangle(resultImg, new Point(r.x,r.y),
                new Point(r.x + r.width - 1, r.y + r.height - 1),
                new Scalar(0,0,255));
        }
        String fileName = Utils.getNonExistingFileName(DETECTIONPATH+"detection_result.png", ".png");
        Imgcodecs.imwrite(fileName, resultImg);
        Utils.showImageInPopup(Utils.matToBufferedImage(resultImg, null));
        return listOfDetections;
    }

    public void loadDetectors()
    {
        for(String label : Constants.LABELS)
        {
            String detectorPath = OBJECT_DETECTOR_FOLDER_PATH+"ObjectDetector_"+label;
            boolean doesDetectorExist =
                    Utils.checkFileExist(detectorPath);
            if(!doesDetectorExist)
                continue;
            SVM labelDetector = SVM.load(detectorPath);
            detectorMap.put(label, labelDetector);
        }
    }

    public static void train(double negativeMaxIoU)
    {
        List<GroundTruth> positiveSamples = DbUtils.readGroundTruths("Label = 0");
//        List<GroundTruth> negativeSamples = DbUtils.readGroundTruths(
//                "Label != 0 AND IoUWithClosestGT < "+negativeMaxIoU);
        List<GroundTruth> negativeSamples = DbUtils.readGroundTruths(
                "(Label = -1 AND IoUWithClosestGT < "+negativeMaxIoU+") OR (Label NOT IN (0,-1))" );
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
