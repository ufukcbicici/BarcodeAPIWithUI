package bandrol_training;

import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Constants {
    public static final int WIDTH = 1000;
    public static final int HEIGHT = 600;
    public static final int X = 0;
    public static final int Y = 1;
    public static final int Z = 2;
    private static final String ROOT_PATH = "C:\\Users\\ufuk.bicici\\Desktop\\Bandrol";
    // public static final String ROOT_PATH = "C:\\Users\\ufuk.bicici\\Desktop\\Bandrol";
    public static final String DEBUGPATH = ROOT_PATH + "\\DebugPics\\";
    public static final String DETECTIONPATH = ROOT_PATH + "\\Detections\\";
    public static final String JSONPATH = ROOT_PATH + "\\Json Annotations\\";
    public static final String LOCALIZED_IMAGE_PATH =  ROOT_PATH + "\\Localized Training Images\\";
    public static final String OBJECT_DETECTOR_FOLDER_PATH = ROOT_PATH + "\\Object Detector  SVMs\\";
    public static final double CHARACTER_LOC_GAUSSIAN_KERNEL = 1;
    public static final double DOWNSIZE_RATIO = 0.5;
//    public static final double REFERENCE_IMAGE_WIDTH = 316;
//    public static final double REFERENCE_IMAGE_HEIGHT = 217;
    public static final Scalar CHROME_GREEN = new Scalar(0, 177, 64);
    //"jdbc:sqlite:C:/sqlite/db/chinook.db";
    public static final String CONNECTION_STRING =
            "jdbc:sqlite:C:/Users/ufuk.bicici/Desktop/Bandrol/bandrol.db";
    public static final String GROUND_TRUTH_TABLE = "BANDROL_GROUND_TRUTHS";
    public static final String HOG_TABLE = "HOG_FEATURES";


    public static final List<String> LABELS;
    static {
        LABELS = new ArrayList<>(Arrays.asList("A","B","C","D","E","F","G","H","I","J","K",
                "L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                "0","1","2","3","4","5","6","7","8","9"));
    }
}
