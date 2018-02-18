package sample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.opencv.core.Core;
import sample.controller.Controller;
import sample.model.Algorithm;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception{
        Algorithm.InitAlgorithm();
        FXMLLoader loader = new FXMLLoader(getClass().getResource("view/sample.fxml"));
        Parent rootBorderPane = loader.load();
        Controller controller = loader.getController();
        primaryStage.setTitle("Hello World");
        primaryStage.setScene(new Scene(rootBorderPane, Constants.WIDTH, Constants.HEIGHT));
        primaryStage.show();
    }


    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
