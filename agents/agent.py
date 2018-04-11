import numpy as np
import tensorflow as tf
from collections import deque

def linterp(arr0, arr1, tau=0.5):
    return (1-tau) * arr0 + tau * arr1

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
                 tau=1e-2
                ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        
        self.gamma = gamma
        self.tau = tau
        
        self.actor_target = NeuralActor(name="actor_target", action_size=self.action_size, state_size=self.state_size, action_low=action_low, action_high=action_high, num_hidden=num_hidden, learning_rate=learning_rate)
        self.critic_target = NeuralCritic(name="critic_target", state_size=self.state_size, action_size=self.action_size, num_hidden=num_hidden, learning_rate=learning_rate, gamma=gamma)
        self.actor_local = NeuralActor(name="actor_local", action_size=self.action_size, state_size=self.state_size, action_low=action_low, action_high=action_high, num_hidden=num_hidden, learning_rate=learning_rate)
        self.critic_local = NeuralCritic(name="critic_local", state_size=self.state_size, action_size=self.action_size, num_hidden=num_hidden, learning_rate=learning_rate, gamma=gamma)
        self.memory = Memory(max_size=memory_size, a=memory_a, eps=memory_eps)
        self.noise = OUNoise(self.action_size, exploration_mu, exploration_theta, exploration_sigma)
        
        self.rewards = []
        self.losses = []
    
    def step(self, sess, state, action, reward, next_state, done):
        self.rewards.append(reward)
        scores, action_gradients = self.critic_target.critique(sess, [state], [action])
        self.memory.add((state, action, reward, next_state, done), score=np.sqrt(np.sum(np.square(action_gradients[0]))))
        if done:
            if self.memory.count() > self.batch_size:
                loss = self.learn(sess)
                self.losses.append(loss)
            self.noise.reset()
    
    def learn(self, sess):
        states, actions, rewards, next_states, episode_ends = zip(*self.memory.sample(self.batch_size))
        
        next_actions = self.actor_target.act(sess, next_states)
        
        next_scores, next_action_gradients = self.critic_target.critique(sess, next_states, next_actions)
        next_scores[[episode_ends]] = 0
        targets = rewards + self.gamma * next_scores
        critic_loss = self.critic_local.learn(sess, states, actions, targets)
        
        scores, action_gradients = self.critic_local.critique(sess, states, actions)
        actor_loss = self.actor_local.learn(sess, states, scores, action_gradients)

        al = self.actor_local.get_weights(sess)
        at = self.actor_target.get_weights(sess)
        at_next = [linterp(at[i], al[i], tau=self.tau) for i in range(len(at))]
        
        cl = self.critic_local.get_weights(sess)
        ct = self.critic_target.get_weights(sess)
        ct_next = [linterp(ct[i], cl[i], tau=self.tau) for i in range(len(ct))]

        self.actor_target.set_weights(sess, *at_next)
        self.critic_target.set_weights(sess, *ct_next)
        
        return actor_loss, critic_loss
    
    def act(self, sess, state):
        return self.actor_target.act(sess, [state])[0] + self.noise.sample()
        
            
class NeuralActor(object):
    def __init__(self, name="actor", state_size=6, action_size=4, action_low=0, action_high=1, num_hidden=64, learning_rate=1e-3):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.hidden0 = tf.contrib.layers.fully_connected(self.state, num_hidden, scope='hidden0',
                                                            weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0, num_hidden, scope='hidden1',
                                                            weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.action_activation = tf.contrib.layers.fully_connected(
                self.hidden1, self.action_size, activation_fn=tf.nn.sigmoid, scope='output',
                weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.action = self.action_range * self.action_activation + self.action_low
            
            self.score = tf.placeholder(tf.float32, [None])
            self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
            self.loss = tf.reduce_mean(-self.action_gradient * self.action)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
            self.train_vars = tf.trainable_variables(scope=self.name)
            self.weights_to_assign = [tf.placeholder(v.dtype, v.shape.as_list()) for v in self.train_vars]
            self.assignments = [tf.assign(self.train_vars[i], self.weights_to_assign[i]) for i in range(len(self.train_vars))]
    
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
    
    def get_weights(self, sess):
        train_vars = tf.trainable_variables(scope=self.name)
        return sess.run(train_vars)
    
    def set_weights(self, sess, *weights_to_assign):
        if len(weights_to_assign) != len(self.weights_to_assign):
            raise Exception("weights_to_assign length %d does not match self.weights_to_assign length %d" % (len(weights_to_assign), len(self.weights_to_assign)))
        feed_dict = dict([(self.weights_to_assign[i], weights_to_assign[i]) for i in range(len(self.weights_to_assign))])
        sess.run(self.assignments, feed_dict=feed_dict)

class NeuralCritic(object):
    def __init__(self, name="critic", state_size=6, action_size=4, num_hidden=16, learning_rate=1e-3, gamma=0.99):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.action = tf.placeholder(tf.float32, [None, self.action_size], name='action')
            self.input_ = tf.concat([self.state, self.action], axis=1)
            
            self.hidden0 = tf.contrib.layers.fully_connected(self.input_, num_hidden, scope='hidden0',
                                                             weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0, num_hidden, scope='hidden1',
                                                             weights_initializer=tf.truncated_normal_initializer(stddev=1e-1))
            self.score = tf.squeeze(tf.contrib.layers.fully_connected(self.hidden1, 1, scope='output',
                                                            weights_initializer=tf.truncated_normal_initializer(stddev=1e-1),
                                                            activation_fn=None))
            self.action_gradient = tf.gradients(self.score, [self.action])[0]
            
            self.target = tf.placeholder(tf.float32, [None])
            self.loss = tf.reduce_mean(tf.square(self.target - self.score))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
            self.train_vars = tf.trainable_variables(scope=self.name)
            self.weights_to_assign = [tf.placeholder(v.dtype, v.shape.as_list()) for v in self.train_vars]
            self.assignments = [tf.assign(self.train_vars[i], self.weights_to_assign[i]) for i in range(len(self.train_vars))]
    
    def learn(self, sess, states, actions, targets):
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
    
    def get_weights(self, sess):
        train_vars = tf.trainable_variables(scope=self.name)
        return sess.run(train_vars)
    
    def set_weights(self, sess, *weights_to_assign):
        if len(weights_to_assign) != len(self.weights_to_assign):
            raise Exception("weights_to_assign length %d does not match self.weights_to_assign length %d" % (len(weights_to_assign), len(self.weights_to_assign)))
        feed_dict = dict([(self.weights_to_assign[i], weights_to_assign[i]) for i in range(len(self.weights_to_assign))])
        sess.run(self.assignments, feed_dict=feed_dict)
                                       
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