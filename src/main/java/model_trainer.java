import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Set;
import java.util.stream.Collectors;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class model_trainer {
    public static void main(String[] args) throws IOException {

        ArrayList<String> fileLists = new ArrayList<>();
        ArrayList<Symbol> symbolList = new ArrayList<>();

       String filepathOfTrainingData= "/Users/ivy/Desktop/Senior_Seminar/manuscript/training-data";

        Files.walk(Paths.get(filepathOfTrainingData))
                .filter(Files::isRegularFile)
                .forEach(n -> fileLists.add(n.toString()));

        fileLists.removeIf(n->n.contains(".md")||n.contains(".DS_Store"));

//        System.out.println(fileLists);
//        for (String fileName: fileLists){
//            symbolList.add(new Symbol(fileName));
//        }

//        ArrayList<String> symbolTypes = (ArrayList<String>) symbolList.stream().map(n->n.getName()).collect(Collectors.toList());

//        System.out.println(symbolTypes);
        final String value = "Hello from " + TensorFlow.version();
        System.out.println(value);

    }
}
