class ProjetsController < ApplicationController
  before_action :set_projet, only: %i[ show update destroy ]

  # GET /projets
  def index
    @projets = Projet.all

    render json: @projets
  end

  # GET /projets/1
  def show
    render json: @projet
  end

  # POST /projets
  def create
    authorize @projet = Projet.new(projet_params)

    if @projet.save
      render json: @projet, status: :created, location: @projet
    else
      render json: @projet.errors, status: :unprocessable_entity
    end
  end

  # PATCH/PUT /projets/1
  def update
    if @projet.update(projet_params)
      render json: @projet
    else
      render json: @projet.errors, status: :unprocessable_entity
    end
  end

  # DELETE /projets/1
  def destroy
    @projet.destroy
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_projet
      authorize @projet = Projet.find(params[:id])
    end

    # Only allow a list of trusted parameters through.
    def projet_params
      params.require(:projet).permit(:titre, :description, :pipeline)
    end
end
