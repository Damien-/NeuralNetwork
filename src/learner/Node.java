package learner;

public abstract class Node {
	private double activation;
	private double delta;
	NeuralNet dis;
	public Branch[] brnI, brnO;
	
	
	public Node(double act, NeuralNet dd)
	{
		activation=act;
		dis=dd;
	}
	public abstract void updateBias(double lrnRate);
	public abstract void sigmoid();
	public abstract void delta();
	//public abstract void activateNode();
	public double getActivation(){return activation;}
	public void setActivation(double set){activation=set;}
	
	public void setDelta(double set){delta=set;}
	public double getDelta(){return delta;}
	
	public double sigmoidPrime(){
		return getActivation()*(1-getActivation());
	}

}
