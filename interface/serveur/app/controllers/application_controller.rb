class ApplicationController < ActionController::API
  include AuthentificationModule
  include Pundit::Authorization

  before_action do |c|
    uri = request.env["REQUEST_URI"]
    authenticate_user unless route_toujours_autorisée?(uri)
  end
  before_action :log_appel
  before_action :set_userstamp

  rescue_from Pundit::NotAuthorizedError, with: :user_not_authorized

  def user_not_authorized
    render json: { exception: "Unauthorized", message:"Vous n'avez pas les droits pour faire ça"}, status: :unauthorized # 401
  end

  def set_userstamp
    User.current_user = User.current
  end

  def route_toujours_autorisée?(uri)
  [
    '/api/login.json',
    '/api/renouvellement-jeton',
  ].include? uri \
    or uri.match(/\/api\/verification-jeton/) \
    or uri.match(/projet/) # TODO Gestion des droits 
  end

  def pundit_user
    User.current
  end
  alias :current_user :pundit_user

  def user_for_paper_trail
    pundit_user&.id_res
  end

  def log_appel
    log = Logger.new('log/appels.log')
    log.info [
      request.env["HTTP_X_CLIENTIP_DMZ"]||request.env["REMOTE_ADDR"],
      current_user&.id_res,
      request.method,
      request.url].join(' ')
    log.close
  end
end
