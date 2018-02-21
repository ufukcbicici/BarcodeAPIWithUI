package bandrol_training.model;

import javafx.scene.shape.Rectangle;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LabelingStateContainer
{
    public static Map<Rectangle, GroundTruth> groundTruthMap;
    public static List<Rectangle> rectangleList;
    public static Mat sourceTrainingImg = null;
    public static Rectangle currBB;
    public static int currentSelectedIndex;

    public static void reset()
    {
        groundTruthMap = new HashMap<>();
        rectangleList = new ArrayList<>();
        sourceTrainingImg = null;
        currBB = null;
        currentSelectedIndex = 0;
    }

    public static int getPrevIndex()
    {
        if(rectangleList.size() == 0)
            return 0;
        else if(currentSelectedIndex - 1 < 0)
            return rectangleList.size() - 1;
        else
            return currentSelectedIndex - 1;
    }

    public static int getNextIndex()
    {
        if(rectangleList.size() == 0)
            return 0;
        else if(currentSelectedIndex + 1 > rectangleList.size() - 1)
            return 0;
        else
            return currentSelectedIndex + 1;
    }
}
