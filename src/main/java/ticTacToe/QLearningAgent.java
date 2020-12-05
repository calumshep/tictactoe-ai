package ticTacToe;

import java.util.ArrayList;
import java.util.List;
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
	int numEpisodes = 60000;

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
		this(new RandomAgent(), 0.1, 60000, 0.9);
	}
	
	/**
	 * Returns the action chosen by the exploit property of the epsilon-greedy policy (max q-value)
	 * 
	 * @param g State to get action out of
	 * @return Move Best action out of specified Game state
	 */
	public Move exploit(Game g)
	{
		Move action = null;
		
		double maxQ = -Double.MAX_VALUE;
		
		for (Move m : qTable.get(g).keySet()) {
			// If actions with the same q-value are encountered, the last encountered one is chosen
			if (qTable.getQValue(g, m) >= maxQ) {
				maxQ = qTable.getQValue(g, m);
				action = m;
			}
		}
		
		return action;
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
				List<Move> possibleActions = new ArrayList<Move>(qTable.get(currentState).keySet());

				// Pick an action out of the state based on epsilon-greedy
				Move action = null;
				if (new Random().nextDouble() <= this.epsilon) {
					// Explore (pick random action)
					action = possibleActions.get(new Random().nextInt(possibleActions.size()));
				} else {
					// Exploit (pick action according to current policy (max q-value))
					action = this.exploit(currentState);
				}
				
				if (action != null) {
					// Move should never be illegal but have to catch the exception to keep Java happy anyway
					try {
						// Execute the chosen move and get the new Game state
						Outcome o = env.executeMove(action);
						
						if (!o.sPrime.isTerminal()) {
							// Get the max q-value of actions out of the new state
							double maxQSPrime = this.qTable.getQValue(o.sPrime, this.exploit(o.sPrime));
							// Work out the 'sample' (scaled reward from action - factored into new q-value)
							double newQ = o.localReward + (this.discount * maxQSPrime);
	
							// Calculate the new q-value and update it
							double currentQ;
							try {
								currentQ = qTable.getQValue(currentState, action);
							} catch (NullPointerException e) {
								currentQ = 0;
							}
							
							// Update the q-value of the action taken out of the state
							System.out.println(((1 - this.alpha) * currentQ) + (this.alpha * newQ));
							qTable.addQValue(
								currentState,
								action,
								((1 - this.alpha) * currentQ) + (this.alpha * newQ)
							);
						}
					} catch (IllegalMoveException e) {
						//continue;
					}
				}
				
				// Repeat for the new state after executing the move
				currentState = env.getCurrentGameState();
			}
			System.out.println("Finished episode " + i);

			env = new TTTEnvironment(currentState.o);
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
		for (Game state : qTable.keySet()) {
			// Work out best action based on highest q-value
			Move bestAction = this.exploit(state);
			
			if (bestAction != null) {
				policy.policy.put(state, bestAction);
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
