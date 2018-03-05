package bandrol_training.model.Detectors;

import bandrol_training.Constants;
import bandrol_training.model.Detection;
import bandrol_training.model.Utils;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public abstract class DetectionMethod {

    protected Table<Integer, Integer, Mat> extractFeatures(Mat img, int sliding_window_width, int sliding_window_height,
                                                           double sourceImgWidth)
    {
        Table<Integer, Integer, Mat> hogTable = HashBasedTable.create();
        HogOpenCVExtractor [] extractors = new HogOpenCVExtractor[Constants.THREAD_COUNT];
        int columnsPerThread = img.cols() / Constants.THREAD_COUNT;
        for(int tid=0;tid<Constants.THREAD_COUNT;tid++)
        {
            int startColumn = tid * columnsPerThread;
            assert startColumn < img.cols();
            int endColumn = (tid < Constants.THREAD_COUNT - 1) ? (startColumn + columnsPerThread - 1) : (img.cols()-1);
            assert endColumn < img.cols();
            assert startColumn <= endColumn;
            extractors[tid] = new HogOpenCVExtractor(startColumn, endColumn, img,
                    sourceImgWidth, sliding_window_width, sliding_window_height);
            extractors[tid].start();
        }
        for(int tid=0;tid<Constants.THREAD_COUNT;tid++)
        {
            try {
                extractors[tid].join();
                hogTable.putAll(extractors[tid].getHogTable());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        return hogTable;
    }

    public abstract void detect(Mat img, int sliding_window_width, int sliding_window_height,
                                double sourceImgWidth, double nms_iou_threshold);

}
