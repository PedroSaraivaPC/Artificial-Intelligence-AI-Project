import numpy as np

from . import intelligence


class gwo(intelligence.sw):
    """
    Grey Wolf Optimizer
    """

    def __init__(self, n, function, lb, ub, dimension, iteration):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        """

        super(gwo, self).__init__()

        self.__agents = np.random.uniform(lb, ub, (n, dimension)) #cria os lobos, n lobos , posicoes aleatorias por causa do random.uniform
        self._points(self.__agents) #guardar as posicoes iniciais
        alpha, beta, delta = self.__get_abd(n, function) #avaliar  os lobos para decidir os liders

        Gbest = alpha #melhor solucao para ja

        for t in range(iteration):

            a = 2 - 2 * t / iteration #a = fator de energia , começa em 2 e desce ate 0 (fim da caça, ataque final)

            r1 = np.random.random((n, dimension))
            r2 = np.random.random((n, dimension))
            A1 = 2 * r1 * a - a #controla se o lobo se afasta ou aproxima
            C1 = 2 * r2  #da um peso aleatorio a presa, para tornar mais natural

            r1 = np.random.random((n, dimension))
            r2 = np.random.random((n, dimension))
            A2 = 2 * r1 * a - a
            C2 = 2 * r2

            r1 = np.random.random((n, dimension))
            r2 = np.random.random((n, dimension))
            A3 = 2 * r1 * a - a
            C3 = 2 * r2

            Dalpha = abs(C1 * alpha - self.__agents) #distancia do alfa
            Dbeta = abs(C2 * beta - self.__agents)
            Ddelta = abs(C3 * delta - self.__agents)

            X1 = alpha - A1 * Dalpha #ir para a beira do alfa
            X2 = beta - A2 * Dbeta
            X3 = delta - A3 * Ddelta

            self.__agents = (X1 + X2 + X3) / 3 #o lobo normal vai para a media dos 3 chefes

            self.__agents = np.clip(self.__agents, lb, ub) #para nao fugir do limite
            self._points(self.__agents)

            alpha, beta, delta = self.__get_abd(n, function) #recalcula os novos liders
            if function(alpha) < function(Gbest):
                Gbest = alpha#se for melhor , atualiza o lider

        self._set_Gbest(Gbest)
        alpha, beta, delta = self.__get_abd(n, function)
        self.__leaders = list(alpha), list(beta), list(delta)

    def __get_abd(self, n, function): #rerturnar quem sao os 3 liders da ronda

        result = []
        fitness = [(function(self.__agents[i]), i) for i in range(n)] #cria uma lista de pares (fitness, indice do lobo)
        
        fitness.sort()#ordena a lista do menor fitness para o maior

        for i in range(3): #os 3 melhores
            result.append(self.__agents[fitness[i][1]])

        return result #devolve as posicoes dos liders

    def get_leaders(self):
        """Return alpha, beta, delta leaders of grey wolfs"""

        return list(self.__leaders)
