package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent
{
	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction = new HashMap<Game, Double>();
	
	/**
	 * The discount factor
	 */
	double discount = 0.9;
	
	/**
	 * The MDP model
	 */
	TTTMDP mdp = new TTTMDP();
	
	/**
	 * The number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k = 10;
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp = new TTTMDP();
		this.discount = 0.9;
		initValues();
		train();
	}
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p)
	{
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor)
	{
		this.discount = discountFactor;
		mdp = new TTTMDP();
		initValues();
		train();
	}

	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount = discountFactor;
		mdp = new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames = Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		
		for(Game g: allGames) {
			this.valueFunction.put(g, 0.0);
		}
	}
	
	/**
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 */
	public void iterate()
	{
		// Perform k number of iterations
		for (int i = 0; i < this.k; i++) {
			// Start at the first state in the valueFunction map
			for (Game currentGame : this.valueFunction.keySet()) {
				// Get all the possible actions from the current state (Game)
				List<Move> currentMoves = currentGame.getPossibleMoves();

				double stateValue;
				
				if (currentGame.isTerminal()) {
					// Terminal state value is always zero
					stateValue = 0.0;
					continue;
				} else {
					/*
					 * MAX_VALUE is largest *magnitude*, so using negative for numerically smallest
					 * Ensures that first comparison for max will work even if new max is negative
					 */
					stateValue = -Double.MAX_VALUE;
				}
				
				// Compute value of current state (Game)
				for (Move move : currentMoves) {
					// Get all the possible outcomes from the current action (Move)
					List<TransitionProb> T = mdp.generateTransitions(currentGame, move);

					// Compute the Bellman equation for this action
					double sum = 0.0;
					for (TransitionProb t : T) {
						sum += (t.prob * (
							t.outcome.localReward + this.discount * this.valueFunction.get(t.outcome.sPrime))
						);
					}

					/*
					 * Check if sum (sPrime value) is max of iterations thus far
					 * (i.e. it is the overall state value)
					 */
					if (sum > stateValue) {
						stateValue = sum;
					}
				}

				// Store the value of the current state (Game)
				this.valueFunction.replace(currentGame, stateValue);
			}
		}
	}
	
	/**
	 * This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		Policy policy = new Policy();
		
		// Start at the first state in the valueFunction map
		for (Game currentGame : this.valueFunction.keySet()) {
			// Get all the possible actions from the current state (Game)
			List<Move> currentMoves = currentGame.getPossibleMoves();
			
			double stateValue;
			
			if (currentGame.isTerminal()) {
				// Terminal state value is always zero
				continue;
			} else {
				/*
				 * MAX_VALUE is largest *magnitude*, so using negative for numerically smallest
				 * Ensures that first comparison for max will work even if new max is negative
				 */
				stateValue = -Double.MAX_VALUE;
			}
			
			for (Move move : currentMoves) {
				// Get all the possible outcomes from the current action (Move)
				List<TransitionProb> T = mdp.generateTransitions(currentGame, move);

				// Compute the Bellman equation for this action
				double sum = 0.0;
				for (TransitionProb t : T) {
					sum += (t.prob * (
						t.outcome.localReward + this.discount * this.valueFunction.get(t.outcome.sPrime))
					);
				}

				// Check if sum (sPrime value) is max thus far (i.e. so far it is the state value)
				if (sum >= stateValue) {
					stateValue = sum;
					policy.policy.put(currentGame, move);
				}
			}
		}
		
		return policy;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		// First run value iteration
		this.iterate();

		/**
		 * Now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 */
		super.policy = this.extractPolicy();
		
		if (this.policy == null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		// Test method to play the agent against a human agent.
		ValueIterationAgent agent = new ValueIterationAgent();
		HumanAgent d = new HumanAgent();
		
		Game g = new Game(agent, d, d);
		g.playOut();
	}
}
