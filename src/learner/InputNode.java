package learner;

public class InputNode extends Node {

//	public Branch[] brnO;
	
	public int brnNo(){return brnO.length;}
	public InputNode(double act, NeuralNet dd){
		super(act,dd);
		brnO = new Branch[dd.getHiddenLayers()[0]];
		
	}
	@Override
	public void sigmoid() {
		System.out.println("N SIGMOID FUNCTION FOR U, input node, has nothing to apply it to");
	}
	@Override
	public void delta() {
		System.out.println("N DELTA FOR U, input node, has none");
	}
	@Override
	public void updateBias(double lrnRate) {
		System.out.println("N BIAS FOR U, input node, has none");
	}

}
