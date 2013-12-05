package learner;

import java.io.IOException;

public class MainArchitecture {

	public static void main(String[] args) throws IOException, InterruptedException {
			NeuralNet net = new NeuralNet(args[0]);
			net.ReadInputNNA();
			net.Architecture();
									
	}

}
