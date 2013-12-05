package learner;

public class OutputNode extends Node {
	private double yAccuali;
//	public Branch[] brnI;
	private double bias = Math.random()/5-0.1;
	public void updateBias(double lrnRate){
	bias+=lrnRate*getDelta();
	}
	public double getBias(){return bias;}
	public void setBias(double set){bias=set;}
	
	public void setYAccuali(double set){yAccuali=set;}
	public double getYAccuali(){return yAccuali;}
	public OutputNode(double act, NeuralNet dd){
		super(act,dd);
		brnI = new Branch[dd.getHiddenLayers()[dd.getHiddenLayersNo()-1]];
	}
	@Override	
	public void sigmoid() {
		 double inp=0;
		 for(int i=0; i<brnI.length;i++)
			 inp+=brnI[i].getWeight()*brnI[i].leftNode().getActivation();
		 inp+=bias;
		 inp=1/(1+Math.pow(2.73, -inp));
		 setActivation(inp);
	}
	@Override
	public void delta() {
		setDelta(sigmoidPrime()*(yAccuali-getActivation()));
	}

}
