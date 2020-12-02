package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an
 * {@link Outcome} object, in other words an [s,a,r,s']: source state, action taken, reward received, and the target state
 * after the opponent has played their move. You may want/need to edit {@link TTTEnvironment} - but you probably won't need to.
 * 
 * @author ae187
 */

public class QLearningAgent extends Agent
{
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha = 0.1;

	/**
	 * The number of episodes to train for
	 */
	int numEpisodes = 100;

	/**
	 * The discount factor (gamma)
	 */
	double discount = 0.9;

	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon = 0.1;

	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair, you can do
	 * qTable.get(game).get(move) which return the Q(game,move) value stored. Be careful with 
	 * cases where there is currently no value. You can use the containsKey method to check if the mapping is there.
	 */
	
	QTable qTable = new QTable();

	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env = new TTTEnvironment();

	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * 
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha = learningRate;
		this.numEpisodes = numEpisodes;
		this.discount = discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 */
	
	protected void initQTable()
	{
		// All valid games where it is X's turn, or it's terminal
		List<Game> allGames = Game.generateAllValidGames('X');
		
		for(Game g: allGames) {
			List<Move> moves = g.getPossibleMoves();
			
			for(Move m: moves) {
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
		}
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set
	 * these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 100, 0.9);
	}
	
	
	/**
	 * Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating
	 * q-values according to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where
	 * with the probability {@code epsilon} the agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 * At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned
	 * q-values. This is currently done for you on the last line of the method.
	 */
	
	public void train()
	{
		// Play the specified number of episodes
		for (int i = 0; i < this.numEpisodes; i++) {
			// Start at the initial game state
			Game currentState = env.getCurrentGameState();
			
			// Continue playing until a terminal state is reached
			while (!currentState.isTerminal()) {
				// Get the actions associated with the state as a List type
				List<Move> possibleActions = currentState.getPossibleMoves();

				// Pick an action out of the state based on epsilon-greedy
				Move action = null;
				if (new Random().nextDouble() <= this.epsilon) {
					// Explore (pick random action)
					action = possibleActions.get(new Random().nextInt(possibleActions.size()));
				} else {
					// Exploit (pick action according to current policy (max q-value))
					double maxQ = 0.0;
					for (Move m : possibleActions) {
						// If actions with the same q-value are encountered, the last encountered one is chosen
						if (qTable.getQValue(currentState, m) >= maxQ) {
							maxQ = qTable.getQValue(currentState, m);
							action = m;
						}
					}
				}
				
				if (action != null) {
					// Move should never be illegal but have to catch the exception to keep Java happy anyway
					try {
						// Execute the chosen move and get the reward
						double sample = env.executeMove(action).localReward + (this.discount * 0);
						// Calculate the new q-value and update it
						qTable.addQValue(
							currentState,
							action,
							((1 - this.alpha) * (qTable.getQValue(currentState, action) + sample))
						);
					} catch (IllegalMoveException e) {
						continue;
					}
				}
			}
		}

		this.policy = this.extractPolicy();

		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/**
	 * Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		Policy policy = new Policy();
		
		// Iterate over every stored state
		for (Entry<Game, HashMap<Move, Double>> stateSet : qTable.entrySet()) {
			// Work out best action based on highest q-value
			Move bestAction = null;
			double maxQVal = 0.0;
			for (Entry<Move, Double> possibleAction : stateSet.getValue().entrySet()) {
				if (possibleAction.getValue() >= maxQVal) {
					maxQVal = possibleAction.getValue();
					bestAction = possibleAction.getKey();
				}
			}
			
			if (bestAction != null) {
				policy.policy.put(stateSet.getKey(), bestAction);
			}
		}

		return policy;
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent = new QLearningAgent();
		
		HumanAgent d = new HumanAgent();
		
		Game g = new Game(agent, d, d);
		g.playOut();
	}
}
