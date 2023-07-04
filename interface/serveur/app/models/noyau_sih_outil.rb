class NoyauSihOutil
  
  def self.demander_infos_utilisateurs_authentification(id_res)
    url = "http://#{NOYAUSIH}/iam/"+id_res+"/#{CI_AGAGOU}/habilitations.json"
    uri = URI.parse(url)
    reponse = Net::HTTP.get_response(uri)
    if reponse.is_a?(Net::HTTPSuccess)
      return JSON.parse(reponse.body)
    else
      self.log_erreur_noyausih reponse, __method__
    end
  end

  private
  def self.log_erreur_noyausih reponse=nil, nom_method
    log = Logger.new('log/demande_noyau_sih.log')
    log.info "#{nom_method} : #{reponse}"
    throw "Erreur demande d\'info à NoyauSIH (#{__method__}) : #{reponse}"
  end

end
