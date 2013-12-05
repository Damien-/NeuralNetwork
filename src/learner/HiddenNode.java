package learner;

public class HiddenNode extends Node {
//	public Branch[] brnI, brnO;
	private int layerNo;
	private double bias = Math.random()/5-0.1;
	public void updateBias(double lrnRate){
		bias+=lrnRate*getDelta();
	}
	public double getBias(){return bias;}
	public void setBias(double set){bias=set;}
	
	public HiddenNode(double act, NeuralNet dd, int lNo){
		super(act,dd); layerNo=lNo;
		int hlp;
		if(layerNo==0) 
			hlp = dis.getInputNo();
		else hlp = dis.getHiddenLayers()[layerNo-1];
		brnI = new Branch[hlp];
		if(dd.getHiddenLayersNo()==layerNo+1) 
			hlp = dis.getOutputNo();
		else hlp = dis.getHiddenLayers()[layerNo+1];
		brnO = new Branch[hlp];
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

	public void delta(){
		double sum=0;
		for(int i=0; i<brnO.length; i++)
			sum+=brnO[i].getWeight()*brnO[i].rightNode().getDelta();
		setDelta(sigmoidPrime()*sum);
	}

	
	//public Branch brnOGet(int i){return brnO[i];}
	//public Branch brnIGet(int i){return brnI[i];}
}
