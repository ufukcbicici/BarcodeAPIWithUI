package bandrol_training.model;

import bandrol_training.Constants;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;

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
        String sql = "INSERT INTO "+Constants.GROUND_TRUTH_TABLE+
                "(FileName,Label,XCoord,YCoord,Width,Height) VALUES(?,?,?,?,?,?);";
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
                prep.addBatch();
            }
            int[] updateCounts = prep.executeBatch();
            conn.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
