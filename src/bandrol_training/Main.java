package bandrol_training;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.opencv.core.Core;
import bandrol_training.controller.Controller;
import bandrol_training.model.Algorithm;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception{
        Algorithm.InitAlgorithm();
        FXMLLoader loader = new FXMLLoader(getClass().getResource("view/bandrol_ui.fxml"));
        Parent rootBorderPane = loader.load();
        Controller controller = loader.getController();
        controller.setPrimaryStage(primaryStage);
        controller.initUIElements();
        primaryStage.setTitle("Hello World");
        primaryStage.setScene(new Scene(rootBorderPane, Constants.WIDTH, Constants.HEIGHT));
        primaryStage.show();
    }


    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
