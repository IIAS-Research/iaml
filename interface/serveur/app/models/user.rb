class User < ApplicationRecord
  has_and_belongs_to_many :profils
  cattr_accessor :current_user

  def to_s
    nom_complet
  end

  def self.current
    Current.user
  end

  def self.current=(user)
    Current.user = user
  end

end
