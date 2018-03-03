package bandrol_training.post_processing;

import bandrol_training.model.DbUtils;
import bandrol_training.model.GroundTruth;
import bandrol_training.model.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.EM;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import java.util.*;

public class ProbabilisticPostProcesser {

    public static void train(int topLeftMixtureCount)
    {
        // Read all ground truth bounding boxes.
        String exclusionStatement = "FileName NOT IN" + Utils.getFileSelectionClause();
        String filterClause = Utils.getFilterClause(
                "Label != -1",
                "ABS(VerticalDisplacement) = 0",
                "ABS(HorizontalDisplacement) = 0",
                "ABS(Rotation) = 0", exclusionStatement);
        List<GroundTruth> unAugmententedSamples = DbUtils.readGroundTruths(filterClause);
        // Group according to files
        Map<String, List<GroundTruth>> fileGTMap = new HashMap<>();
        for(GroundTruth gt : unAugmententedSamples)
        {
            if(!fileGTMap.containsKey(gt.fileName))
                fileGTMap.put(gt.fileName, new ArrayList<>());
            fileGTMap.get(gt.fileName).add(gt);
        }
        Mat topLeftBBCoords = new Mat(0, 2, CvType.CV_32F);
        // Detect the top left character's bounding box for every training file.
        for(String fileName : fileGTMap.keySet())
        {
            List<GroundTruth> gtListForFile = fileGTMap.get(fileName);
            assert gtListForFile.size() % 14 == 0;
            // First sort vertically, split upper and lower rows.
            gtListForFile.sort(Comparator.comparingInt(gt0 -> gt0.getBoundingRect().y));
            // The upper half of the list belongs to upper rows.
            List<GroundTruth> upperRow  = gtListForFile.subList(0, gtListForFile.size()/2);
            List<GroundTruth> lowerRow  = gtListForFile.subList(gtListForFile.size()/2, gtListForFile.size());
            // Sort the upper row horizontally
            upperRow.sort(Comparator.comparingInt(gt -> gt.getBoundingRect().x));
            // Get the top left bounding box. If there is multiple annotations for the same file, there can be
            // multiple coords.
            Mat upperLeftCoord = new Mat(1, 2, CvType.CV_32F);
            upperLeftCoord.put(0,0, upperRow.get(0).getBoundingRect().x);
            upperLeftCoord.put(0,1, upperRow.get(0).getBoundingRect().y);
            topLeftBBCoords.push_back(upperLeftCoord);
        }
        // Learn a Gaussian Mixture Model to model the top left bounding box' coordinate distribution.
        TermCriteria terminationCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                1000, 1e-4 );
        EM em = EM.create();//new EM(topLeftMixtureCount, EM.COV_MAT_DIAGONAL, terminationCriteria);
        em.setTermCriteria(terminationCriteria);
        em.setClustersNumber(topLeftMixtureCount);
        System.out.println(topLeftBBCoords.dump());
        TrainData trainData = TrainData.create(topLeftBBCoords, Ml.ROW_SAMPLE, new Mat());
        em.train(trainData, EM.COV_MAT_GENERIC);
        Mat weights = em.getWeights();
        Mat means = em.getMeans();
        System.out.println(weights.dump());
        System.out.println(means.dump());
        System.out.println("XXX");
    }
}
