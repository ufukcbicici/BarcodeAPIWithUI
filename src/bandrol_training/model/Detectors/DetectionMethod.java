package bandrol_training.model.Detectors;

import bandrol_training.Constants;
import bandrol_training.model.HOGExtractor;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.opencv.core.Mat;

public abstract class DetectionMethod {

    protected Table<Integer, Integer, Mat> extractFeatures(Mat img, int sliding_window_width, int sliding_window_height,
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
                Mat hogFeature = HOGExtractor.extractOpenCVHogFeature(imgRect, sliding_window_width,
                        sliding_window_height);
                hogTable.put(i,j,hogFeature);
            }
        }
        return hogTable;
    }

    public abstract void detect(Mat img, int sliding_window_width, int sliding_window_height,
                                double sourceImgWidth);

}
