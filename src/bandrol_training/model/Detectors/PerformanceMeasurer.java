package bandrol_training.model.Detectors;

import bandrol_training.Constants;
import bandrol_training.model.DbUtils;
import bandrol_training.model.Detection;
import bandrol_training.model.GroundTruth;
import bandrol_training.model.Utils;
import org.apache.commons.math3.util.Pair;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.print.attribute.ResolutionSyntax;
import java.io.File;
import java.util.*;

public class PerformanceMeasurer {

    private static Map<String, List<GroundTruth>> groundTruthMap;
    static
    {
        groundTruthMap = new HashMap<>();
    };

    public static void measure(DetectionMethod detectionMethod,
                               int sliding_window_width,
                               int sliding_window_height,
                               double referenceWidth,
                               double nms_iou_threshold,
                               String testFolder)
    {
        Map<String, Double> truePositives = new HashMap<>();
        Map<String, Double> falsePositives = new HashMap<>();
        Map<String, Double> falseNegatives = new HashMap<>();
        for(String label : Constants.CURR_LABELS)
        {
            truePositives.put(label, 0.0);
            falsePositives.put(label, 0.0);
            falseNegatives.put(label, 0.0);
        }
        List<File> listOfFiles = Utils.getAllFilesUnderFolder(testFolder);
        for(File file : listOfFiles)
        {
            if(!file.isFile())
                continue;
            String fileName = file.getName();
            String filePath = file.getPath();
            List<GroundTruth> groundTruthList;
            Mat image = Imgcodecs.imread(filePath, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            assert image.rows() > 0 && image.cols() > 0;
            Mat resizedSource = new Mat();
            double resizeRatio = referenceWidth / image.cols();
            Imgproc.resize(image, resizedSource,
                    new Size(resizeRatio*image.cols(),resizeRatio*image.rows()));
            if(!groundTruthMap.containsKey(fileName))
            {
                String filterClause = Utils.getFilterClause(
                        "FileName = "+"\""+fileName+"\"",
                        "Label != -1",
                        "ABS(VerticalDisplacement) = 0",
                        "ABS(HorizontalDisplacement) = 0",
                        "ABS(Rotation) = 0");
                groundTruthList = DbUtils.readGroundTruths(filterClause);
                groundTruthMap.put(fileName, groundTruthList);
            }
            else
            {
                groundTruthList = groundTruthMap.get(fileName);
            }
            assert groundTruthList.size() == 14;
            List<Detection> detectionList = detectionMethod.detect(resizedSource, sliding_window_width,
                    sliding_window_height, referenceWidth, nms_iou_threshold, false);
            // Intersections with the ground truths; this determines true and false negatives.
            Set<GroundTruth> usedGroundTruths = new HashSet<>();
            Set<Detection> usedDetections = new HashSet<>();
            for(GroundTruth currGroundTruth : groundTruthList)
            {
                Rect groundTruthRect = currGroundTruth.getBoundingRect();
                boolean isDetected = false;
                for(Detection detection : detectionList)
                {
                    Rect detectionRect = detection.getRect();
                    double iou = Utils.calculateIoU(groundTruthRect, detectionRect);
                    if(iou >= Constants.POSITIVE_IOU_THRESHOLD)
                    {
                        if(currGroundTruth.label.equals(detection.getLabel()))
                        {
                            if(!isDetected)
                            {
                                isDetected = true;
                                truePositives.put(currGroundTruth.label, truePositives.get(currGroundTruth.label)+1);
                            }
                            else
                            {
                                falsePositives.put(currGroundTruth.label, falsePositives.get(currGroundTruth.label)+1);
                            }
                        }
                        else
                        {
                            falsePositives.put(detection.getLabel(), falsePositives.get(detection.getLabel())+1);
                        }
                        usedDetections.add(detection);
                        usedGroundTruths.add(currGroundTruth);
                    }
                }
            }
            for(GroundTruth groundTruth : groundTruthList)
            {
                if(!usedGroundTruths.contains(groundTruth))
                {
                    falseNegatives.put(groundTruth.label, falseNegatives.get(groundTruth.label)+1);
                }
            }
            for(Detection detection : detectionList)
            {
                if(!usedDetections.contains(detection))
                    falsePositives.put(detection.getLabel(), falsePositives.get(detection.getLabel())+1);
            }
        }
        System.out.println("XXX");
    }

}
