module AuthentificationModule
  require "net/http"

  DUREE_DE_VIE_JETON = 8.hours
  DEBUG_AUTH = false

  def noyausih_url
    NOYAUSIH
  end

  def create_or_update_user(payload)

    # On remet le courriel dnas le payload au cas où NoyauSIH ne le renvoie pas (mais on l'aurait par les agents)
    payload["courriel"] ||= User.where("lower(windows) = lower(?)", payload["windows"]).first&.courriel
    h = {
      nom_complet: payload[:nom_complet],
      courriel: payload[:courriel],
      windows: payload[:sub],
      id_res: payload[:id_res],
    }
    # User.lock :
    # permet d'éviter qu'en cas de requêtes concurrentes
    # qui arrivent en même temps,
    #  on crée plusieurs lignes identifiques
    # dans la table malgré le validates :uniqueness
    u = User.where("lower(windows) = lower(?)", h[:windows]).first
    if u.present?
      u.update(h)
    else
      u = User.lock.create(h)
    end
    return u
  end

  # GET /api/login.json
  def create_jwt impersonate: nil
    log = Logger.new('log/jwt.log')
    log.info "Demande de JWT"
    if params[:ticket].present? and params[:service].present?
      log.debug "ticket et service sont présents"
      retour_cas = cas_verify ticket: params[:ticket], service: params[:service]
      l = Logger.new('log/jwt_debug.log')
      l.debug params
      l.debug retour_cas
      l.close
      # viter les appels qui n'ont pas permis de vérifier le ST sur CAS
      if retour_cas.first == "yes"
        user = retour_cas.last
      else
        render json: {exception: "Pas les droits", message: "Vous n'avez accès à cette ressource"}, status: 401 # Unauthorized
        return
      end
    end

    if impersonate && User.current.admin
      user = User.find_by(id_res: impersonate).windows
    end

    unless user 
      render json: {exception: "User not found", message: "Utilisateur pas trouver"}, status: 500 # Unauthorized
    end

    puts "User: #{user}" if DEBUG_AUTH
    iat = Time.now  # Issued At Time
    exp = iat + DUREE_DE_VIE_JETON   # Expiration
    user_db = User.find_by(id_res: user)

    infos_utilisateur = NoyauSihOutil.demander_infos_utilisateurs_authentification(user)
    attributs_supplementaires = {
      id_res: user,
      courriel: infos_utilisateur["personne"]["courriel"],
      nom_complet: infos_utilisateur["personne"]["nom"] + " " + infos_utilisateur["personne"]["prenom"]
    }

    payload = {sub: user, aud: nil, iat: iat.to_i, exp: exp.to_i}.merge attributs_supplementaires
    secret = Rails.application.credentials.jwt_secret
    raise Exception.new "Le secret n'est pas défini (rails credentials:edit)" unless secret
    log.debug "payload : #{payload}"
    u = create_or_update_user payload
    log.debug "u : #{u}"

    render json: {token: JWT.encode(payload, secret, 'HS256')}
  ensure
    log.close
  end

  def authenticate_user
    User.current = nil
    log = Logger.new('log/jwt.log')
    log.info "authenticate_user"
    if request.env["HTTP_AUTHORIZATION"] and request.env["HTTP_AUTHORIZATION"] != "undefined" or params["token"].present?
      token = request.env["HTTP_AUTHORIZATION"]&.gsub("Bearer ","") || params["token"]
    end
    if token
      begin
        jwt = verifie_jwt token
      rescue JWT::ExpiredSignature => e
        render json: {exception: e.class.to_s, message: e}, status: 401
        return
      end
      if jwt
        log.debug "On a un JWT, on cherche le user #{jwt.first["sub"]}"
        u = User.where("lower(windows) = lower(?)", jwt.first["sub"]).first
        User.current = u
      end
    else
      log.debug "pas de token"
    end
    unless u
      log.debug "pas d'utilisateur trouvé"
      render json: {exception: "PasConnecté", message: "Vous n'avez accès à cette ressource"}, status: 401
    end
    nil
  ensure
    log&.close
  end

  def renouvellement_jeton
    verify_authenticity_token force_regeneration: true
    render json: Time.now
  end

  def verification_jeton
    log = Logger.new("log/verification_jeton.log")
    m = request.headers["HTTP_X_ORIGINAL_URI"]&.match(/jeton=([^&]+)/)
    if m
      jwt = m[1]
    else
      jwt = params["jeton"]
    end
    log.debug jwt

    if jwt
      verifie_jwt(jwt)
      log.info "ok"
      head :ok
    else
      head :forbidden
    end
  rescue Exception => e
    log.info "ko"
    head :forbidden
  end

  private
  def verifie_jwt token
    puts token if DEBUG_AUTH
    jwt = JWT.decode token, Rails.application.credentials.jwt_secret, true, { algorithm: 'HS256' }
    if jwt and jwt.first["sub"].present?
      token = jwt.first
      puts Time.strptime(token["iat"].to_s, '%s').in_time_zone('CET').to_s if DEBUG_AUTH
      puts Time.strptime(token["exp"].to_s, '%s').in_time_zone('CET').to_s if DEBUG_AUTH
    end
    jwt if jwt and jwt.first["sub"].present?
  end

  def verify_authenticity_token force_regeneration: false
    # En mode test, ne pas vérifier l'authentification pour le moment
    return true if Rails.env.test?

    uri = request.env["REQUEST_URI"]
    if !route_toujours_autorisée?(uri) or force_regeneration
      token = verifie_jwt(request.env["HTTP_AUTHORIZATION"]&.gsub("Bearer", "") || params["token"])&.first
      # Si on a un token et qu'on a dépassé la moitié de sa durée de vie, on le regénère
      if force_regeneration or (token and (Time.now.to_i > token["iat"] + (token["exp"] - token["iat"])/2))
        secret = Rails.application.credentials.jwt_secret
        iat = Time.now  # Issued At Time
        exp = iat + DUREE_DE_VIE_JETON   # Expiration
        token[:iat] = iat.to_i  # Issued At Time
        token[:exp] = exp.to_i   # Expiration
        response.set_header "NEW_TOKEN", JWT.encode(token, secret, 'HS256')
      end
    end
  rescue Exception => e
    puts e.message.red
    render json: {exception: e.class.to_s, message: e.message}, status: 500
    # raise e
  end

  # Vérication que le couple (ticket, service) est bien valide auprès du serveur CAS
  def cas_verify(ticket: nil, service: nil)
    # Effacer le User.current pour que si on échoue, ça ne soit pas au nom de quelqu'un d'autre
    User.current = nil
    cas_server = YAML.load_file("config/cas.yml")["cas"]
    url = [cas_server, "validate?service=#{service}&ticket=#{ticket}"].join("/")
    puts url if DEBUG_AUTH
    if Rails.env.development? and ENV['U']
      puts "Simulation de l'utilisateur #{ENV['U']}"
      return ['yes', ENV['U']]
    end

    reponse = Net::HTTP.get_response(URI.parse(url))
    # Renvoie un objet avec status, plein de choses
    # CAS renvoie :
    # yes ou no \n
    # le compte Windows \n
    reponse.body.split("\n")
  end

end
