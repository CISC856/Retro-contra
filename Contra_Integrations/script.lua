previous_x = 0
previous_score = 0
previous_scroll =0
previous_life = 2
function update_x()
    local x_reward=0
    local score_reward=0
    local scroll_reward=0
    local death_reward=0
    local defeat_reward=0
    if data.x_pos > previous_x then
        x_reward = data.x_pos - previous_x
        previous_x = data.x_pos
    elseif data.x_pos < previous_x then
        x_reward = data.x_pos - previous_x
        previous_x = data.x_pos
    else
    	x_reward = 0
    end
    if data.score > previous_score then
        score_reward = (data.score-previous_score)*0.0001
        previous_score = data.score
    else
    	score_reward = 0
    end
    if data.scroll_center > previous_scroll then
        scroll_reward = data.scroll_center-previous_scroll
        previous_scroll = data.scroll_center
    else
    	scroll_reward = 0
    end
    if data.life < previous_life then
    	death_reward=-25
    	previous_life = data.life
    else 
    	death_reward=0
    end
    if data.boss_defeat== 8 then
    	defeat_reward=1000
    else
    	defeat_reward=0
    end
    local result = math.min(math.max(-15,x_reward),15) + score_reward +scroll_reward + death_reward
    return result
end 
