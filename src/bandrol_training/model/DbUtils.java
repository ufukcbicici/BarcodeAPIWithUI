package bandrol_training.model;

import bandrol_training.Constants;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.sql.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class DbUtils {

    private static Connection connect()
    {
        Connection conn = null;
        try
        {
            // create a connection to the database
            conn = DriverManager.getConnection(Constants.CONNECTION_STRING);
            System.out.println("Connection to SQLite has been established.");
            return conn;
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
        return null;
    }

    public static void writeGroundTruth(List<GroundTruth> groundTruthList)
    {
        Connection conn = connect();
        String sql = "INSERT INTO "+Constants.HOG_TABLE+
                "(FileName,Label,XCoord,YCoord,Width,Height,Rotation,VerticalDisplacement,HorizontalDisplacement," +
                "IoUWithClosestGT,HOGFeature) VALUES(?,?,?,?,?,?,?,?,?,?,?);";
        try
        {
            assert conn != null;
            conn.setAutoCommit(false);
            PreparedStatement prep = conn.prepareStatement(sql);
            for(GroundTruth gt : groundTruthList)
            {
                prep.setString(1, gt.fileName);
                prep.setString(2, gt.label);
                prep.setDouble(3, gt.x);
                prep.setDouble(4, gt.y);
                prep.setDouble(5, gt.width);
                prep.setDouble(6, gt.height);
                prep.setDouble(7, gt.rotation);
                prep.setDouble(8, gt.verticalDisplacement);
                prep.setDouble(9, gt.horizontalDisplacement);
                prep.setDouble(10, gt.iouWithClosestGroundTruth);
                prep.setBytes(11, gt.getHogFeatureAsByteArr());
                prep.addBatch();
            }
            int[] updateCounts = prep.executeBatch();
            int totalNumOfUpdates = Arrays.stream(updateCounts).sum();
            if(totalNumOfUpdates != groundTruthList.size())
                throw new SQLException("Number of updates do match!");
            conn.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public static List<GroundTruth> readGroundTruths(String filterClause)
    {
        Connection conn = connect();
        String sql = "SELECT * FROM "+Constants.HOG_TABLE;
        if(filterClause != null && !filterClause.isEmpty())
            sql += " WHERE " + filterClause;
        List<GroundTruth> resultList = new ArrayList<>();
        try {
            assert conn != null;
            Statement stmt  = conn.createStatement();
            ResultSet rs    = stmt.executeQuery(sql);
            while (rs.next())
            {
                //FileName,Label,XCoord,YCoord,Width,Height,IoUWithClosestGT,HOGFeature
                String fileName = rs.getString("FileName");
                String label = rs.getString("Label");
                int xCoord = rs.getInt("XCoord");
                int yCoord = rs.getInt("YCoord");
                int width = rs.getInt("Width");
                int height = rs.getInt("Height");
                double rotation = rs.getDouble("Rotation");
                double verticalDisplacement = rs.getDouble("VerticalDisplacement");
                double horizontalDisplacement = rs.getDouble("HorizontalDisplacement");
                double iou = rs.getDouble("IoUWithClosestGT");
                byte [] hogArr = rs.getBytes("HOGFeature");
                GroundTruth gt = new GroundTruth(fileName, label, xCoord, yCoord, width, height, iou);
                gt.setAugmentationParams(rotation, verticalDisplacement, horizontalDisplacement);
                if(hogArr != null && hogArr.length > 0)
                {
                    int featureDimension = hogArr.length / 8;
                    double [] doubleArr = new double[featureDimension];
                    ByteBuffer bb = ByteBuffer.wrap(hogArr);
                    for (int i = 0; i < featureDimension; i++)
                    {
                        doubleArr[i] = bb.getDouble();
                    }
                    Mat hogVector = new Mat(featureDimension, 1, CvType.CV_64F);
                    hogVector.put(0,0, doubleArr);
                    gt.setHogFeature(hogVector);
                }
                resultList.add(gt);
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return resultList;
    }

    public static void writeToKVMatrixTable(String matrixName, Mat matrix)
    {
        Connection conn = connect();
        String sql = "INSERT INTO "+Constants.MATRIX_KV_TABLE+
                "(MatrixName,Row,Column,Value) VALUES(?,?,?,?);";
        try
        {
            assert conn != null;
            conn.setAutoCommit(false);
            PreparedStatement prep = conn.prepareStatement(sql);
            for(int i=0;i<matrix.rows();i++)
            {
                for(int j=0;j<matrix.cols();j++)
                {
                    prep.setString(1, matrixName);
                    prep.setInt(2, i);
                    prep.setInt(3, j);
                    prep.setDouble(4,matrix.get(i,j)[0]);
                    prep.addBatch();
                }
            }
            int[] updateCounts = prep.executeBatch();
            int totalNumOfUpdates = Arrays.stream(updateCounts).sum();
            if(totalNumOfUpdates != matrix.rows()*matrix.cols())
                throw new SQLException("Number of updates do match!");
            conn.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public static void writeToObjectDetectionSvmTable(String svmFileName, double positiveSign,
                                                      String label, int isActive)
    {
        Connection conn = connect();
        String sql = "INSERT INTO "+Constants.OBJECT_DETECTOR_SVM_TABLE+
                "(SVMFileName,PositiveLabelSign,DetectionLabel,IsActive) VALUES(?,?,?,?);";
        try
        {
            assert conn != null;
            conn.setAutoCommit(false);
            PreparedStatement prep = conn.prepareStatement(sql);
            prep.setString(1, svmFileName);
            prep.setDouble(2, positiveSign);
            prep.setString(3, label);
            prep.setInt(4,isActive);
            prep.addBatch();
            int[] updateCounts = prep.executeBatch();
            int totalNumOfUpdates = Arrays.stream(updateCounts).sum();
            if(totalNumOfUpdates != 1)
                throw new SQLException("Number of updates do match!");
            conn.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public static Mat readFromKVMatrixTable(String matrixName)
    {
        Table<Integer, Integer, Double> intermediateTable = HashBasedTable.create();
        Connection conn = connect();
        String sql = "SELECT * FROM "+Constants.MATRIX_KV_TABLE+" WHERE MatrixName = "+matrixName;
        int maxRow = -1;
        int maxCol = -1;
        Mat matrix = null;
        try {
            assert conn != null;
            Statement stmt  = conn.createStatement();
            ResultSet rs    = stmt.executeQuery(sql);
            while (rs.next())
            {
                int row = rs.getInt("Row");
                int col = rs.getInt("Column");
                double value = rs.getDouble("Value");
                intermediateTable.put(row,col,value);
                if(row > maxRow)
                    maxRow = row;
                if(col > maxCol)
                    maxCol = col;
            }
            if(maxRow < 0 || maxCol < 0)
                throw new SQLException("Degenerate matrix!");
            matrix = Mat.zeros(maxRow+1,maxCol+1,CvType.CV_64F);
            for(Table.Cell<Integer,Integer,Double> cell : intermediateTable.cellSet())
                matrix.put(cell.getRowKey(), cell.getColumnKey(), cell.getValue());

        } catch (SQLException e) {
            e.printStackTrace();
        }
        assert matrix != null;
        return matrix;
    }
}
