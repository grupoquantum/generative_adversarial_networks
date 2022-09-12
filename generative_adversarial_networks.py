from Neuraline.ArtificialIntelligence.DeepLearning.generative_adversarial_networks import GenerativeAdversarialNetworks
generative_adversarial_networks = GenerativeAdversarialNetworks()
print('[treinando o modelo, aguarde...]')
result1 = generative_adversarial_networks.addFit(url_path='./images/samples')
result2 = generative_adversarial_networks.saveModel('gans_paisagens')
if result1 and result2: print('[...modelo treinado com sucesso]')
else: print('[erro no treinamento do modelo.]')