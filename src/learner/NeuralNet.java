package learner;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Random;
import java.util.Scanner;
public class NeuralNet {
	private String driver;
	private String IFile;
	
	private int hiddenLayersNo;
	private int[] hiddenLayers = new int[10];
	//NN number of layers and vector with number of nodes in each of them
	
	private double alfa;
	private double error_tolerance;
	private double scale;
	private int inputNo, outputNo;
	private int n; // number of total I/Os
	private Scanner file;
	private double pom;
	
	private int binNo; //bin number and 
	private Bin[] bins; //bin vectors containing equal parts of I/Os
	
	private static final int archL=2; //defines max layers when testing for different architectures with driver
	private static final int archN=5; //defines max nodes --^
	
	private int[] bestArch;
	private double bestArchError;
	private Bin testBin; //current bin being tested
	private double archError;
	
	public int getInputNo(){return inputNo;}
	public int getOutputNo(){return outputNo;}
	
	private static final double tolerance = 0.000001; 
	//tolerance to smallest change in error rate during the NN construction, smaller it is, less iterations there are
	
	private InputNode[] inputNodes;
	private OutputNode[] outputNodes;
	private Node[] hlpHiddenNodes, prevHiddenNodes;
	
	public int getHiddenLayersNo(){return hiddenLayersNo;}
	public int[] getHiddenLayers(){return hiddenLayers;}
	
	public double getAlfa(){return alfa;}
	public double getErrorTol(){return error_tolerance;}
	public double getScale(){return scale;}
	
	private Elem head;

	public Elem Head(){return head;}
	public void Head(Elem dis){head=dis;}
	public boolean HeadNULL(){ return head==null;}
	
	public class Elem {
		private double[] X, actualY;
	    Elem sled = null;
	    Elem(String file) {
	    	n++;
	    	X = new double[inputNo];
	    	actualY= new double[outputNo];	
	    	String[] arrs = file.split(",");
	    	for(int i=0; i<inputNo; i++)
	    		X[i]=Double.parseDouble(arrs[i]);
	    	for(int i=inputNo; i<outputNo+inputNo; i++)
	    		actualY[i-inputNo]=Double.parseDouble(arrs[i]);
	    	
			if (HeadNULL()){ head = this; return;}
			Elem tek=Head(); 
			while(tek.sled!=null)
				tek=tek.sled;
			tek.sled = this; 
	    }
	}
	public void ReadInputNN() throws IOException{
		file = new Scanner(new File(driver));
		IFile = file.next();
		hiddenLayersNo = file.nextInt();
		for(int i=0; i<hiddenLayersNo; i++) hiddenLayers[i] = file.nextInt();
		alfa = file.nextDouble();
		error_tolerance = file.nextDouble();
		file.close();
		readInputValues();
		
	}
	public void ReadInputNNA() throws IOException{
		file = new Scanner(new File(driver));
		IFile = file.next();
		binNo = file.nextInt();
		alfa = file.nextDouble();
		error_tolerance = file.nextDouble();
		file.close();
		readInputValues();
		
	}
	private void readInputValues() throws IOException
	{
		BufferedReader br = new BufferedReader(
				new InputStreamReader(new FileInputStream(IFile)));
		String line;
		line = br.readLine();
		scale=Double.parseDouble(line);
		line = br.readLine();
		String[] arr = line.split(",");
		inputNo = Integer.parseInt(arr[0]);
		outputNo = Integer.parseInt(arr[1]);
		while((line=br.readLine())!=null)
		new Elem(line);
		file.close();
	}
	public void printIO(){
		String s = "I:";
		for(int i=0; i<inputNo; i++)
			s+=" "+inputNodes[i].getActivation();
		s+="\nO:";
		for(int i=0; i<outputNo; i++)
			s+=" "+outputNodes[i].getYAccuali()*scale;
		s+="\nH:";
		for(int i=0; i<outputNo; i++){
			s+=" "+outputNodes[i].getActivation()*scale;
		}
		System.out.print(s+"\n\n");
		
	}
	public void printO(){
		String s="\nO:";
		for(int i=0; i<outputNo; i++)
			s+=" "+outputNodes[i].getYAccuali()*scale;
		s+="\nH:";
		for(int i=0; i<outputNo; i++){
			s+=" "+outputNodes[i].getActivation()*scale;
		}
		System.out.print(s+"\n");
		
	}
	public NeuralNet(String s) throws IOException {
		driver = s;
	}
	public void updateIO(Elem tek)
	{
		for(int i = 0; i < inputNo; i++)
			inputNodes[i].setActivation(tek.X[i]);
		for(int i = 0; i < outputNo; i++)
			outputNodes[i].setYAccuali(tek.actualY[i]/scale);
	}
	public double updateError(){
		double sum=0;
		for(int i=0; i<outputNodes.length; i++)
			sum += Math.abs(outputNodes[i].getActivation()-outputNodes[i].getYAccuali());
		return sum;
	}
	public void propagateBackNN() throws IOException{
		double sum = 0;
		constructNet(head);
		do{
			sum=0;
			Elem tek=head;
			while(tek!=null){
				updateIO(tek);
				Activation();
				Deltas();
				Weights();
				printIO();
				sum += updateError();
				tek=tek.sled;
			}
			//System.out.print("Error: "+sum+"\n");
			if(pom==sum) 
				{ System.out.print("Best I can do to match error is "+pom+", you could try different architecture."); break;}
			pom=sum;
		}
		while (sum>error_tolerance);
		System.out.print("Error: "+sum);
	}	
	public void propagateBackNNA(Bin tst) throws IOException{
		double sum = 0;
		int i = 0;
		Elem tek;
		if(tst==bins[0]){
			constructNet(bins[1].binHead);
			tek = bins[1].binHead;
			i=1;
			}
		else { 
			updateNet(bins[0].binHead); 
			tek = bins[0].binHead;
			} //updateNet instead of updateIO in order to reset outputs, makes it slower, worse performance, bigger error for ~5-8 %
		sum=0;
		do{
			sum=0;
			while(i<binNo-1){
				while(tek!=null){
					updateIO(tek);
					Activation();
					Deltas();
					Weights();
					//printIO();
					sum += updateError();
					tek=tek.sled;
				}
				if(i+1>=binNo) break;
				tek=bins[++i].binHead;
				if(tek==tst.binHead) 
					if(i+1>=binNo) break;
					else tek=bins[++i].binHead;
			}
			if(Math.abs(pom-sum)<tolerance) 
			{ 
				//System.out.print("Best I can do to match error is "+pom+", you could try different architecture.\n"); 
				break;
			}
			pom=sum;
			//System.out.print(pom+"\n");
			i=0;
		}
		while (sum>error_tolerance);
		
		//VALIDATION
		sum=0;
		tek=tst.binHead;
		while(tek!=null){
			updateIO(tek);
			Activation();
			printO();
			sum += updateError();
			tek=tek.sled;
		}
		archError+=sum;
	}	
	
	public void constructNet(Elem tek){
			//Set INPUT OUTPUT LAYERS
			inputNodes = new InputNode[tek.X.length];
			outputNodes = new OutputNode[tek.actualY.length];
			for(int i = 0; i < tek.X.length; i++)
				inputNodes[i]= new InputNode(tek.X[i],this);
			for(int i = 0; i < tek.actualY.length; i++){
				outputNodes[i]= new OutputNode(0,this);
				outputNodes[i].setYAccuali(tek.actualY[i]/scale);
			}
			//SET FIRST HIDDEN LAYER THAT IS ASSUMED
			prevHiddenNodes= new HiddenNode[hiddenLayers[0]];
			for(int k = 0; k<hiddenLayers[0]; k++)
				prevHiddenNodes[k] = new HiddenNode(0,this,0);
					
			
			for(int i = 0; i < inputNodes.length; i++)
				for(int j=0; j<prevHiddenNodes.length; j++){
					inputNodes[i].brnO[j] = new Branch(inputNodes[i],1);
					inputNodes[i].brnO[j].setParent(prevHiddenNodes[j], 0);
					prevHiddenNodes[j].brnI[i] =  inputNodes[i].brnO[j]; 
				}
			/*for(int j=0; j<prevHiddenNodes.length; j++){
				prevHiddenNodes[j].sigmoid();
			}*/
			int k;
			//set REst of hidden layers		
			for(k = 1; k<hiddenLayersNo; k++){
				hlpHiddenNodes= new HiddenNode[hiddenLayers[k]];
				for(int j=0; j<hlpHiddenNodes.length; j++)
					hlpHiddenNodes[j] = new HiddenNode(0, this, k);
				for(int i = 0; i < prevHiddenNodes.length; i++)
					for(int j=0; j<hlpHiddenNodes.length; j++){
						prevHiddenNodes[i].brnO[j] = new Branch(prevHiddenNodes[i],1);
						prevHiddenNodes[i].brnO[j].setParent(hlpHiddenNodes[j], 0);
						hlpHiddenNodes[j].brnI[i] =  prevHiddenNodes[i].brnO[j]; 
					}				
				prevHiddenNodes=hlpHiddenNodes;
			/*	for(int j=0; j<prevHiddenNodes.length; j++){
					prevHiddenNodes[j].sigmoid();
				}*/
			}
			if(hiddenLayersNo==k) {
				for(int i = 0; i < prevHiddenNodes.length; i++)
					for(int j=0; j<outputNodes.length; j++){
						prevHiddenNodes[i].brnO[j] = new Branch(prevHiddenNodes[i],1);
						prevHiddenNodes[i].brnO[j].setParent(outputNodes[j], 0);
						outputNodes[j].brnI[i] =  prevHiddenNodes[i].brnO[j]; 
					}
			}
			/*for(int j=0; j<outputNodes.length; j++){
				outputNodes[j].sigmoid();
			}
			/*
			for(int i = 0; i < inputNodes.length; i++){
				System.out.print(inputNodes[i].getActivation());
				for(int j = 0; j < inputNodes[i].brnO.length; j++)
					System.out.print("\t" + inputNodes[i].brnO[j].getWeight());
			}
			System.out.print("\n");
			break;*/
	}
	public void updateNet(Elem tek){
		//Set INPUT OUTPUT LAYERS
		for(int i = 0; i < inputNo; i++)
			inputNodes[i].setActivation(tek.X[i]);
		for(int i = 0; i < outputNo; i++){
			outputNodes[i].setYAccuali(tek.actualY[i]/scale);
			outputNodes[i].setActivation(0);
		}
		
	
}
	public void Activation(){
		for(int i=0; i<inputNodes[0].brnO.length; i++)
			inputNodes[0].brnO[i].rightNode().sigmoid();
		Node tek= inputNodes[0].brnO[0].rightNode();
		for(int j=0; j<hiddenLayersNo; j++){
			for(int k=0; k<tek.brnO.length;k++){
				tek.brnO[k].rightNode().sigmoid();
			}
			tek=tek.brnO[0].rightNode();
		}
	}	
	public void Deltas(){
		for(int i=0; i<outputNodes.length; i++){
			outputNodes[i].delta();
		}
		Node tek= outputNodes[0]; //.brnI[0].leftNode()
		for(int j=0; j<hiddenLayersNo; j++){
			for(int i=0; i<tek.brnI.length; i++)
				tek.brnI[i].leftNode().delta();
			tek=tek.brnI[0].leftNode();
		}		
	}
	public void Weights(){
		for(int i=0; i<inputNodes.length; i++)
			for(int j=0; j<inputNodes[i].brnO.length; j++){
			//	System.out.print(inputNodes[i].brnO[j].getWeight()+"\t");
				inputNodes[i].brnO[j].updateWeight(alfa);
			//	System.out.print(inputNodes[i].brnO[j].getWeight()+"\n");
			}
		//System.out.print("\n");
		hlpHiddenNodes = new HiddenNode[inputNodes[0].brnO.length];
		for(int j=0; j<hlpHiddenNodes.length; j++)
			hlpHiddenNodes[j]=inputNodes[0].brnO[j].rightNode();	
		
		for(int i = 0; i <hiddenLayersNo; i++){
			for(int j=0; j<hlpHiddenNodes.length; j++){
				hlpHiddenNodes[j].updateBias(alfa);
				for(int h=0; h<hlpHiddenNodes[0].brnO.length; h++){
				//	System.out.print(hlpHiddenNodes[j].brnO[h].getWeight()+"\t");
					hlpHiddenNodes[j].brnO[h].updateWeight(alfa);
				//	System.out.print(hlpHiddenNodes[j].brnO[h].getWeight()+"\n");
				}
				//System.out.print("\n");
			}
			if(i==hiddenLayersNo-1) break;
			prevHiddenNodes = new HiddenNode[hlpHiddenNodes[0].brnO.length];
			for(int j=0; j<prevHiddenNodes.length; j++)
				prevHiddenNodes[j]=hlpHiddenNodes[0].brnO[j].rightNode();
			hlpHiddenNodes=prevHiddenNodes;
		}
		for(int j=0; j<outputNodes.length; j++)
			outputNodes[j].updateBias(alfa);
	}	
	public class Bin{
		Elem binHead;
		int itemNo;
		Bin(){
			itemNo=0;
			binHead=null;
		}
		
	}
	void sortBins(){
		Random pom = new Random();
		Elem tek=head;
		while(tek!=null){
			int hlp=pom.nextInt(binNo);
			if(bins[hlp].itemNo<=n/binNo){
				Elem binHlp= bins[hlp].binHead;
				bins[hlp].binHead=tek;
				tek=tek.sled;
				bins[hlp].binHead.sled=binHlp;
				bins[hlp].itemNo++;
			}
		}
	}
	public void Architecture() throws IOException{
		bins=new Bin[binNo];
		for(int i=0; i<binNo; i++)
			bins[i]=new Bin();
		sortBins();
		for(int i=0; i<archL; i++){
			hiddenLayersNo=i+1;			
			updateArchNodes(hiddenLayersNo-1);
		}
		System.out.print("\nBest architecture:\n");
		for(int l=0; l < bestArch.length; l++) System.out.print(bestArch[l]+" ");
		statistics(bestArchError);
	}
	public void updateArchNodes(int k) throws IOException{
		for(int j=1; j<=archN; j++){
			hiddenLayers[k]=j;
			if(k==0)
				//for(int l=0; l < hiddenLayersNo; l++) System.out.print(hiddenLayers[l]);
			testArchitecture(); //in if 
			if(k-1>=0) 
				updateArchNodes(k-1);
		}
	}
	public void testArchitecture() throws IOException{
		//rotate bin	
		archError=0;
		for(int k=0; k<binNo; k++){
			testBin=bins[k];
			//System.out.print("Testing bin: "+k+"\n");
			propagateBackNNA(testBin);
		}
		String s="";
		for(int l=0; l < hiddenLayersNo; l++) s+=hiddenLayers[l]+" ";
		System.out.print("Validate arch "+s+" with error:"+archError/(binNo-1)+"\n");
		//for(int l=0; l < hiddenLayersNo; l++) System.out.print(hiddenLayers[l]+" ");
		//System.out.print("-- Arch error: "+archError/binNo+"  --\n");
		if(archError<bestArchError || bestArchError==0){ 
			bestArchError=archError; 
			bestArch=new int[hiddenLayersNo]; 
			for(int i=0; i<hiddenLayersNo; i++) 
				bestArch[i]=hiddenLayers[i];
		}
		
	}
	public void statistics(double dbl){
		    FileWriter fWriter = null;
		    BufferedWriter writer = null;
		    try {
		    fWriter = new FileWriter("statistics.txt",true);
		    writer = new BufferedWriter(fWriter);
		    String s="";
		    for(int l=0; l < bestArch.length; l++) s+=bestArch[l]+" ";
		    	writer.write(s+" "+dbl/10);
		    	writer.newLine();
		    writer.close();
		    } catch (Exception e) {
		    }
		}
	public double bestArch() {
		return bestArchError;
	}
}