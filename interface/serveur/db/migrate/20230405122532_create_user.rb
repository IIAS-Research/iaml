class CreateUser < ActiveRecord::Migration[7.0]
  def change
    create_table :users do |t|
      t.string :nom_complet
      t.string :courriel
      t.string :windows, null: false
      t.string :id_res

      t.timestamps
    end
  end
end
