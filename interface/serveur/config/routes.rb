Rails.application.routes.draw do
  resources :projets
  scope :api do

    # Authentication
    post 'login', to: 'application#create_jwt'

  end
end
