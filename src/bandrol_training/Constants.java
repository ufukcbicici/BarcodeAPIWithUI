package bandrol_training;

import org.opencv.core.Scalar;

import java.util.*;

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
    public static final String CLASSIFIER_SVM_PATH = ROOT_PATH + "\\Classifier SVMs\\";
    public static final String TRAINING_IMAGES = ROOT_PATH + "\\Training Samples\\";
    public static final String TEST_IMAGES = ROOT_PATH + "\\Test Samples\\";
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
    public static final double QR_RATIO = 0.1;


    public static final List<String> LABELS;
    public static final List<String> CURR_LABELS;
    public static final Map<String, Integer> CHAR_TO_LABEL_MAP;
    public static final Map<Integer, String> LABEL_TO_CHAR_MAP;
    public static final List<String> NOISY_FILES;

    static {
        LABELS = new ArrayList<>(Arrays.asList("0","1","2","3","4","5","6","7","8","9",
                "A","B","C","D","E","F","G","H","I","J","K",
                "L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z" ));
        CURR_LABELS = new ArrayList<>(Arrays.asList("0","1","2","3","4","5","6","7","8","9",
                "A","B","C","D","E","F","G","H","J","K","M","N","P","Q","R","S","T","V","W","X","Y","Z" ));
        CHAR_TO_LABEL_MAP = new HashMap<>();
        LABEL_TO_CHAR_MAP = new HashMap<>();
        for(int i=0; i < LABELS.size(); i++) {
            CHAR_TO_LABEL_MAP.put(LABELS.get(i),i);
            LABEL_TO_CHAR_MAP.put(i, LABELS.get(i));
        }

        NOISY_FILES = Arrays.asList("localized_40.png");
    }

    // Bu ayarları cross validation ile bulduk.
    public static final Map<String, Double> BEST_NMS_IOU_THRESHOLDS;
    static
    {
        BEST_NMS_IOU_THRESHOLDS = new HashMap<String, Double>();
        BEST_NMS_IOU_THRESHOLDS.put("0", 0.1);
    }
}
