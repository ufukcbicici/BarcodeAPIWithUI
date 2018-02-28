package bandrol_training.model;

import bandrol_training.Constants;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.util.ArrayList;
import java.util.List;

public class SVMEnsemble {
    private List<SVM> svmList;

    public SVMEnsemble()
    {
        this.svmList = new ArrayList<>();
    }

    public SVMEnsemble(List<SVM> svmList)
    {
        this.svmList = svmList;
    }

    public List<SVM> getSvmList()
    {
        return svmList;
    }

    public void add(SVM svm)
    {
        svmList.add(svm);
    }

    public Mat predictByVoting(Mat samples)
    {
        //svm.predict(hog32f, response, StatModel.RAW_OUTPUT);
        Mat voteTable = Mat.zeros(samples.rows(), Constants.LABELS.size(), CvType.CV_32S);
        for(SVM svm : svmList) {
            Mat responses = new Mat();
            svm.predict(samples, responses, 0);
            for(int i=0;i<responses.rows();i++)
            {
                int responseLabel = (int)responses.get(i,0)[0];
                int labelVoteCount = (int)voteTable.get(i, responseLabel)[0];
                voteTable.put(i, responseLabel, labelVoteCount+1);
            }

        }
        Mat finalResponses = new Mat(samples.rows(), 1, CvType.CV_32S);
        for(int i=0;i<samples.rows();i++)
        {
            Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(voteTable.row(i));
            int label = (int)minMaxLocResult.maxLoc.x;
            finalResponses.put(i, 0, label);
        }
        return finalResponses;
    }

    public List<Mat> predictLabels(Mat samples)
    {
        List<Mat> responses = new ArrayList<>();
        for(SVM svm : svmList) {
            Mat response = new Mat();
            svm.predict(samples, response, 0);
            responses.add(response);
        }
        return responses;
    }

    public List<Mat> predictMargins(Mat samples)
    {
        List<Mat> margins = new ArrayList<>();
        for(SVM svm : svmList) {
            Mat response = new Mat();
            svm.predict(samples, response, StatModel.RAW_OUTPUT);
            margins.add(response);
        }
        return margins;
    }
}
