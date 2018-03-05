package bandrol_training.model.Detectors;

import bandrol_training.Constants;
import bandrol_training.model.HOGExtractor;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class HogOpenCVExtractor extends Thread {

    private int startColumn;
    private int endColumn;
    private Mat sourceImg;
    private double sourceImgWidth;
    private int sliding_window_width;
    private int sliding_window_height;
    private Table<Integer, Integer, Mat> hogTable;

    public HogOpenCVExtractor(int startColumn,
                              int endColumn,
                              Mat sourceImg,
                              double sourceImgWidth,int sliding_window_width,int sliding_window_height)
    {
        this.startColumn = startColumn;
        this.endColumn = endColumn;
        this.sourceImg = sourceImg;
        this.sourceImgWidth = sourceImgWidth;
        this.sliding_window_width = sliding_window_width;
        this.sliding_window_height = sliding_window_height;
    }

    public void run() {
        hogTable = HashBasedTable.create();
        double upperRatio = sourceImgWidth * Constants.QR_RATIO;
        for(int i=0;i<sourceImg.rows();i++) {
            for (int j = startColumn; j <= endColumn; j++) {
                if (i < upperRatio || i + sliding_window_height - 1 >= sourceImg.rows())
                    continue;
                if (j + sliding_window_width - 1 >= sourceImg.cols())
                    continue;
                Mat imgRect = sourceImg.submat(i, i + sliding_window_height, j, j + sliding_window_width);
                Mat hogFeatureT = HOGExtractor.extractOpenCVHogFeature(imgRect, sliding_window_width,
                        sliding_window_height);
                Mat hogFeature = new Mat();
                Core.transpose(hogFeatureT, hogFeature);
                Mat hogFeature32f = new Mat();
                hogFeature.convertTo(hogFeature32f, CvType.CV_32F);
                hogTable.put(i, j, hogFeature32f);
            }
        }
    }

    public int getStartColumn() {
        return startColumn;
    }

    public int getEndColumn() {
        return endColumn;
    }

    public Mat getSourceImg() {
        return sourceImg;
    }

    public Table<Integer, Integer, Mat> getHogTable() {
        return hogTable;
    }
}
