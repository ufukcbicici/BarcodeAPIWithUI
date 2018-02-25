package bandrol_training.model;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static bandrol_training.Constants.DEBUGPATH;
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
                                                double nms_iou_threshold)
    {
        Mat resultImg = img.clone();
        SVM svm;
        if(preLoadedSvm==null)
            svm = SVM.load(OBJECT_DETECTOR_FOLDER_PATH+"ObjectDetector");
        else
            svm = preLoadedSvm;
        List<Detection> listOfDetections = new ArrayList<>();
        for(int i=0;i<img.rows();i++)
        {
            for(int j=0;j<img.cols();j++)
            {
                if(i + (int)sliding_window_height - 1 >= img.rows())
                    continue;
                if(j + (int)sliding_window_width - 1 >= img.cols())
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
                double signedDistance = response.get(0,0)[0];
                if(signedDistance < 0)
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
                new Scalar(255,255,0));
        }
        Imgcodecs.imwrite(DEBUGPATH+"detection_result.png", resultImg);
        return maxima;
    }

    public static void train(double negativeMaxIoU)
    {
        List<GroundTruth> positiveSamples = DbUtils.readGroundTruths("Label != -1");
        List<GroundTruth> negativeSamples = DbUtils.readGroundTruths(
                "Label == -1 AND IoUWithClosestGT < "+negativeMaxIoU);
        //List<GroundTruth> negativeSamples = DbUtils.readGroundTruths("Label == -1 AND IoUWithClosestGT < "+negativeMaxIoU);
        System.out.println(positiveSamples.size());
        System.out.println(negativeSamples.size());

        SVM svm = SVM.create();
        TermCriteria terminationCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                1000, 1e-3 );
        svm.setKernel(SVM.LINEAR);

        // Prepare the training set
        int totalSampleCount = positiveSamples.size() + negativeSamples.size();
        int featureDim = positiveSamples.get(0).getHogFeature().rows();
        Mat trainingSamples = new Mat(totalSampleCount, featureDim, CvType.CV_64F);
        Mat labelMat = new Mat(totalSampleCount, 1, CvType. CV_32SC1);
        int [] labelArr = new int[totalSampleCount];
        List<Mat> transposeFeatures = new ArrayList<>();
        for(int i=0;i<positiveSamples.size();i++)
        {
            labelArr[i] = 1;
            Mat hogFeature = positiveSamples.get(i).getHogFeature();
            Mat hogFeatureT = new Mat();
            Core.transpose(hogFeature, hogFeatureT);
            transposeFeatures.add(hogFeatureT);
        }
        for(int i=0;i<negativeSamples.size();i++)
        {
            labelArr[positiveSamples.size()+i] = -1;
            Mat hogFeature = negativeSamples.get(i).getHogFeature();
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
