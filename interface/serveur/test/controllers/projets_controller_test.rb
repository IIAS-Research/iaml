require "test_helper"

class ProjetsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @projet = projets(:one)
  end

  test "should get index" do
    get projets_url, as: :json
    assert_response :success
  end

  test "should create projet" do
    assert_difference("Projet.count") do
      post projets_url, params: { projet: { description: @projet.description, pipeline: @projet.pipeline, titre: @projet.titre } }, as: :json
    end

    assert_response :created
  end

  test "should show projet" do
    get projet_url(@projet), as: :json
    assert_response :success
  end

  test "should update projet" do
    patch projet_url(@projet), params: { projet: { description: @projet.description, pipeline: @projet.pipeline, titre: @projet.titre } }, as: :json
    assert_response :success
  end

  test "should destroy projet" do
    assert_difference("Projet.count", -1) do
      delete projet_url(@projet), as: :json
    end

    assert_response :no_content
  end
end
