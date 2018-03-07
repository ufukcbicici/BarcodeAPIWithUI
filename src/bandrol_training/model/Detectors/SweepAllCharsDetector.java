package bandrol_training.model.Detectors;

import bandrol_training.Constants;
import bandrol_training.model.Algorithms.NonMaximaSuppression;
import bandrol_training.model.ClassifierType;
import bandrol_training.model.Detection;
import bandrol_training.model.Ensembles.EnsembleModel;
import bandrol_training.model.Ensembles.SVMEnsemble;
import bandrol_training.model.Utils;
import com.google.common.collect.Table;
import jdk.jshell.spi.ExecutionControl;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static bandrol_training.Constants.DETECTIONPATH;

public class SweepAllCharsDetector extends DetectionMethod {

    private Map<String, EnsembleModel> ensembleMap;
    private ClassifierType classifierType;

    public SweepAllCharsDetector(ClassifierType classifierType)
    {
        ensembleMap = new HashMap<>();
        this.classifierType = classifierType;
    }

    public void init() throws ExecutionControl.NotImplementedException {
        for(String label : Constants.CURR_LABELS)
        {
            switch (this.classifierType)
            {
                case SVM:
                    SVMEnsemble svmEnsemble = new SVMEnsemble(false);
                    svmEnsemble.loadEnsemble(label);
                    ensembleMap.put(label, svmEnsemble);
                    break;
                case MLP:
                    throw new ExecutionControl.NotImplementedException("MLP not implemented");
            }
        }
    }

    private List<Detection> detectAllChars(
            Table<Integer, Integer, Mat> featureTable,
            int sliding_window_width, int sliding_window_height,
            double nms_iou_threshold)
    {
        SweepDetector [] detectors = new SweepDetector[Constants.THREAD_COUNT];
        List<Detection> allDetections = new ArrayList<>();
        for(int tid=0;tid<Constants.THREAD_COUNT;tid++)
        {
            detectors[tid] = new SweepDetector(ensembleMap, featureTable, sliding_window_width,
                    sliding_window_height, nms_iou_threshold);
        }
        int currIndex = 0;
        for(String label : Constants.CURR_LABELS)
        {
            if (!ensembleMap.containsKey(label))
                continue;
            detectors[currIndex % Constants.THREAD_COUNT].addCharacter(label);
            currIndex++;
        }
        for(int tid=0;tid<Constants.THREAD_COUNT;tid++)
        {
            detectors[tid].start();
        }
        for(int tid=0;tid<Constants.THREAD_COUNT;tid++)
        {
            try {
                detectors[tid].join();
                allDetections.addAll(detectors[tid].getDetections());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        return allDetections;
    }

    public List<Detection>  detect(Mat img, int sliding_window_width, int sliding_window_height,
                                double sourceImgWidth, double nms_iou_threshold, boolean isVerbose)
    {
        long t0 = System.nanoTime();
        Mat canvasImg = img.clone();
        Table<Integer, Integer, Mat> featureTable =
                extractFeatures(img, sliding_window_width, sliding_window_height, sourceImgWidth);
        List<Detection> listOfMaximaForAllLabels = new ArrayList<>();
        long t1 = System.nanoTime();
        System.out.println("Feature Extraction took:"+(double)(t1-t0)/1000000.0+" ms");

        long t2 = System.nanoTime();
        listOfMaximaForAllLabels = detectAllChars(featureTable, sliding_window_width, sliding_window_height,
                nms_iou_threshold);
        long t3 = System.nanoTime();
        System.out.println("Detection Took:"+(double)(t3-t2)/1000000.0+" ms");

        long t4 = System.nanoTime();
        List<Detection> ultimateMaxima = NonMaximaSuppression.run(listOfMaximaForAllLabels, nms_iou_threshold);
        long t5 = System.nanoTime();
        System.out.println("Final NMS Took:"+(double)(t5-t4)/1000000.0+" ms");

        for (Detection dtc : ultimateMaxima) {
            Rect r = dtc.getRect();
            Imgproc.rectangle(canvasImg, new Point(r.x, r.y),
                    new Point(r.x + r.width - 1, r.y + r.height - 1),
                    new Scalar(0, 0, 255));
            int font = Core.FONT_HERSHEY_PLAIN;
            Imgproc.putText(canvasImg, dtc.getLabel(), new Point(r.x + 5, r.y + 10), font,
                    1.0,new Scalar(0,255,0),1);
        }
        if(isVerbose)
        {
            String fileName = Utils.getNonExistingFileName(DETECTIONPATH +
                    "final_detection", ".png");
            Imgcodecs.imwrite(fileName, canvasImg);
            Utils.showImageInPopup(Utils.matToBufferedImage(canvasImg, null));
        }
        return ultimateMaxima;
    }
}
