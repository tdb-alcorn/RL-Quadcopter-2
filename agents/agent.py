import numpy as np
import tensorflow as tf
from collections import deque

class NeuralAgent(object):
    def __init__(self,
                 state_size=6,
                 action_size=4,
                 action_low=0,
                 action_high=1,
                 num_hidden=64,
                 learning_rate=1e-3,
                 memory_size=10000,
                 batch_size=20,
                 gamma=0.99,
                 exploration_mu=0,
                 exploration_theta=0.15,
                 exploration_sigma=0.2,
                 memory_a=0.9,
                 memory_eps=1e-3,
                 l2_regularization=1e-1
                ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        
        self.actor = NeuralActor(action_size=self.action_size, state_size=self.state_size, action_low=action_low, action_high=action_high, num_hidden=num_hidden, learning_rate=learning_rate)
        self.critic = NeuralCritic(state_size=self.state_size, action_size=self.action_size, num_hidden=num_hidden, learning_rate=learning_rate, gamma=gamma)
        self.memory = Memory(max_size=memory_size, a=memory_a, eps=memory_eps)
        self.noise = OUNoise(self.action_size, exploration_mu, exploration_theta, exploration_sigma)
        
        self.rewards = []
        self.losses = []
    
    def step(self, sess, state, action, reward, next_state, done):
        self.rewards.append(reward)
        scores, action_gradients = self.critic.critique(sess, [state], [action])
        self.memory.add((state, action, reward, next_state, done), score=np.sqrt(np.sum(np.square(action_gradients[0]))))
        if done:
            if self.memory.count() > self.batch_size:
                loss = self.learn(sess)
                self.losses.append(loss)
            self.noise.reset()
    
    def learn(self, sess):
        states, actions, rewards, next_states, episode_ends = zip(*self.memory.sample(self.batch_size))
        next_actions = self.actor.act(sess, next_states)
        critic_loss = self.critic.learn(sess, states, actions, rewards, next_states, next_actions, episode_ends)
        scores, action_gradients = self.critic.critique(sess, states, actions)
        actor_loss = self.actor.learn(sess, states, scores, action_gradients)
        return actor_loss, critic_loss
    
    def act(self, sess, state):
        return self.actor.act(sess, [state])[0] + self.noise.sample()
        
            
class NeuralActor(object):
    def __init__(self, state_size=6, action_size=4, action_low=0, action_high=1, num_hidden=64, learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        with tf.variable_scope("actor"):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.hidden0 = tf.contrib.layers.fully_connected(self.state, num_hidden,
                                                            weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0, num_hidden,
                                                            weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.action_activation = tf.contrib.layers.fully_connected(
                self.hidden1, self.action_size, activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.action = self.action_range * self.action_activation + self.action_low
            
            self.score = tf.placeholder(tf.float32, [None])
            self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
            self.loss = tf.reduce_mean(-self.action_gradient * self.action)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    
    def learn(self, sess, states, scores, action_gradients):
        loss, _ = sess.run([self.loss, self.opt], feed_dict={
            self.state: states,
            self.score: scores,
            self.action_gradient: action_gradients,
        })
        return loss
    
    def act(self, sess, state):
        action = sess.run(self.action, feed_dict={
            self.state: state,
        })
        return action

class NeuralCritic(object):
    def __init__(self, state_size=6, action_size=4, num_hidden=16, learning_rate=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        with tf.variable_scope("critic"):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.action = tf.placeholder(tf.float32, [None, self.action_size], name='action')
            self.input_ = tf.concat([self.state, self.action], axis=1)
            
            self.hidden0 = tf.contrib.layers.fully_connected(self.input_, num_hidden,
                                                             weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0, num_hidden,
                                                             weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.score = tf.squeeze(tf.contrib.layers.fully_connected(self.hidden1, 1,
                                                            weights_initializer=tf.truncated_normal_initializer(stddev=1e-1),
                                                            activation_fn=None))
            self.action_gradient = tf.gradients(self.score, [self.action])[0]
            
            self.target = tf.placeholder(tf.float32, [None])
            self.loss = tf.reduce_mean(tf.square(self.target - self.score))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    
    def learn(self, sess, states, actions, rewards, next_states, next_actions, episode_ends):
        next_scores = sess.run(self.score, feed_dict={
            self.state: next_states,
            self.action: next_actions,
        })
        next_scores[[episode_ends]] = 0
        targets = rewards + self.gamma * next_scores
        loss, _ = sess.run([self.loss, self.opt], feed_dict={
            self.state: states,
            self.action: actions,
            self.target: targets,
        })
        return loss
    
    def critique(self, sess, state, action):
        score, action_gradients = sess.run([self.score, self.action_gradient], feed_dict={
            self.state: state,
            self.action: action,
        })
        return score, action_gradients
                                       
class Memory(object):
    def __init__(self, max_size=1000, a=0.9, eps=1e-3):
        self.a = a
        self.eps = eps
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience, score=None):
        score = 0 if score is None else score
        self.buffer.append((experience, score))
            
    def sample(self, batch_size):
        probs = np.array([score + self.eps for _, score in self.buffer])
        probs_pow = probs ** self.a
        probs = probs_pow / np.sum(probs_pow)
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size,
                               p=probs,
                               replace=False)
        return [self.buffer[ii][0] for ii in idx]
    
    def count(self):
        return len(self.buffer)
    
class OUNoise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state