package learner;

import java.io.IOException;

public class Main {

	public static void main(String[] args) throws IOException, InterruptedException {
		NeuralNet net = new NeuralNet(args[0]);
		net.ReadInputNN(); 
		net.propagateBackNN();
			
		/*	
		NeuralNet net = new NeuralNet(args[1]);
		net.ReadInputNNA(); 
		net.Architecture();
		//*/
	}

}
