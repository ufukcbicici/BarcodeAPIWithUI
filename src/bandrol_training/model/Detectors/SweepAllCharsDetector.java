package bandrol_training.model.Detectors;

import bandrol_training.Constants;
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

    public void init(int ensembleCount) throws ExecutionControl.NotImplementedException {
        for(String label : Constants.CURR_LABELS)
        {
            switch (this.classifierType)
            {
                case SVM:
                    SVMEnsemble svmEnsemble = new SVMEnsemble();
                    svmEnsemble.loadEnsemble(ensembleCount, label);
                    ensembleMap.put(label, svmEnsemble);
                    break;
                case MLP:
                    throw new ExecutionControl.NotImplementedException("MLP not implemented");
            }
        }
    }

    public void detect(Mat img, int sliding_window_width, int sliding_window_height,
                                double sourceImgWidth)
    {
        Mat canvasImg = img.clone();
        Table<Integer, Integer, Mat> featureTable =
                extractFeatures(img, sliding_window_width, sliding_window_height, sourceImgWidth);
        List<Detection> listOfDetections = new ArrayList<>();
        for(String label : Constants.CURR_LABELS) {
            if (!ensembleMap.containsKey(label))
                continue;
            EnsembleModel ensembleModel = ensembleMap.get(label);
            for(Table.Cell c : featureTable.cellSet())
            {
                Mat feature = new Mat();
                Core.transpose((Mat) c.getValue(), feature);
                Mat feature32f = new Mat();
                feature.convertTo(feature32f, CvType.CV_32F);


            }
            // ensembleModel.
            // ensembleModel.d
        }

//        Table<Integer, Integer, Mat> featureTable = extractFeatures(
//                img,
//                sliding_window_width,
//                sliding_window_height,
//                source_width);
//        List<Detection> listOfDetections = new ArrayList<>();
//        for(Table.Cell c : featureTable.cellSet())
//        {
//            Mat hogFeatureT = new Mat();
//            Core.transpose((Mat) c.getValue(), hogFeatureT);
//            Mat hog32f = new Mat();
//            hogFeatureT.convertTo(hog32f, CvType.CV_32F);
//            List<Mat> predictedLabels = svmEnsemble.predictLabels(hog32f);
//            List<Mat> predictedMargins = svmEnsemble.predictMargins(hog32f);
//            double totalMarginResponse = 0.0;
//            double totalVote = 0.0;
//            for(int svmIndex=0;svmIndex<svmEnsemble.getSvmList().size();svmIndex++)
//            {
//                double predictedLabel = predictedLabels.get(svmIndex).get(0,0)[0];
//                totalVote += predictedLabel;
//                totalMarginResponse += Math.abs(predictedMargins.get(svmIndex).get(0,0)[0])*predictedLabel;
//            }
//            double avgMarginResponse = totalMarginResponse / (double)svmEnsemble.getSvmList().size();
//            if(totalVote > 0)
//            {
//                Detection detection = new Detection(
//                        new Rect((int) c.getColumnKey(), (int) c.getRowKey(),
//                                sliding_window_width, sliding_window_height), avgMarginResponse);
//                listOfDetections.add(detection);
//            }
//        }
//        List<Detection> maxima = nonMaximaSuppression(listOfDetections, nms_iou_threshold);
//        for (Detection dtc : maxima) {
//            Rect r = dtc.getRect();
//            Imgproc.rectangle(resultImg, new Point(r.x, r.y),
//                    new Point(r.x + r.width - 1, r.y + r.height - 1),
//                    new Scalar(0, 0, 255));
//            int font = Core.FONT_HERSHEY_COMPLEX;
//            DecimalFormat df2 = new DecimalFormat(".##");
//            String doubleFormatted = df2.format(dtc.getResponse());
//            Imgproc.putText(resultImg, doubleFormatted, new Point(r.x, r.y), font,
//                    0.4,new Scalar(0,255,0),1);
//        }
//        String fileName = Utils.getNonExistingFileName(DETECTIONPATH + "detection_result.png", ".png");
//        Imgcodecs.imwrite(fileName, resultImg);
//        Utils.showImageInPopup(Utils.matToBufferedImage(resultImg, null));
//        return maxima;




    }
}
