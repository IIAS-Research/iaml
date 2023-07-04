Rails.application.routes.draw do
  scope :api do

    # Authentication
    post 'login', to: 'application#create_jwt'

  end
end
