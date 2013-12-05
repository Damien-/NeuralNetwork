package learner;

public class Branch {
	private Node left, right;
	private double weight;
	
	public double getWeight(){return weight;}
	public void setWeight(double set){weight=set;}
	public Node leftNode(){return left;}
	public Node rightNode(){return right;}
	
	public Branch(Node parent, int lr)
	{
		if(lr!=0) left=parent;
		else right=parent;
		weight = Math.random()/5-0.1;
	}
	public void setParent(Node parent, int lr)
	{
		if(lr!=0) left=parent;
		else right=parent;
	}
	public void updateWeight(double lrnRate){
		weight+=lrnRate*left.getActivation()*right.getDelta(); 
	}
	
}
