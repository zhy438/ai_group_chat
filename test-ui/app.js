const API_BASE = 'http://localhost:8000/api/v1';

// State
let currentGroupId = null;
let currentGroup = null;
let editingMemberId = null;
let models = [];
let abortController = null;

// DOM Elements
const groupListEl = document.getElementById('groupList');
const messageAreaEl = document.getElementById('messageArea');
const memberListEl = document.getElementById('memberList');
const memberModelSelect = document.getElementById('memberModel');

const chatHeaderEl = document.getElementById('chatHeader');
const memberSectionEl = document.getElementById('memberSection');
const discussionPanelEl = document.getElementById('discussionPanel');
const welcomeScreenEl = document.querySelector('.welcome-screen');

// Initialize
async function init() {
    await loadModels();
    await loadGroups();
    setupEventListeners();
}

// ============ API Calls ============

async function api(path, method = 'GET', body = null) {
    const options = { method, headers: {} };
    if (body) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
    }
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'API请求失败');
    }
    return res.json();
}

async function loadModels() {
    try {
        models = await api('/models');
        memberModelSelect.innerHTML = models.map(m =>
            `<option value="${m.model_id}">${m.name} (${m.context_window / 1000}k)</option>`
        ).join('');
    } catch (e) {
        console.error('加载模型失败:', e);
    }
}

async function loadGroups() {
    try {
        const groups = await api('/groups');
        renderGroupList(groups);
    } catch (e) {
        alert('加载群聊失败: ' + e.message);
    }
}

async function loadGroupDetails(groupId) {
    try {
        const group = await api(`/groups/${groupId}`);
        currentGroupId = groupId;
        currentGroup = group;
        renderGroupDetails(group);
        await loadMessages(groupId);

        // Show UI elements
        chatHeaderEl.style.display = 'flex';
        memberSectionEl.style.display = 'block';
        discussionPanelEl.style.display = 'block';

        // Load context stats
        await loadContextStats(groupId);

        // Sync Dropdown
        const modeSelect = document.getElementById('discussionModeSelect');
        if (modeSelect) {
            modeSelect.value = group.discussion_mode;
        }

        // Sync Compression Threshold Slider
        const slider = document.getElementById('thresholdSlider');
        const sliderVal = document.getElementById('thresholdValue');
        if (slider) {
            const val = group.compression_threshold !== undefined ? group.compression_threshold : 0.8;
            slider.value = val;
            if (sliderVal) sliderVal.textContent = `${(val * 100).toFixed(0)}%`;
        }

        if (welcomeScreenEl) welcomeScreenEl.style.display = 'none';

        // Highlight sidebar item
        document.querySelectorAll('.group-item').forEach(el => {
            el.classList.toggle('active', el.dataset.id === groupId);
        });
    } catch (e) {
        alert('加载群聊详情失败: ' + e.message);
    }
}

async function loadMessages(groupId) {
    try {
        const messages = await api(`/groups/${groupId}/messages`);
        renderMessages(messages);
    } catch (e) {
        console.error('加载消息失败:', e);
    }
}

// ============ UI Rendering ============

function renderGroupList(groups) {
    if (groups.length === 0) {
        groupListEl.innerHTML = '<p class="empty-hint">暂无群聊</p>';
        return;
    }

    groupListEl.innerHTML = groups.map(g => `
        <div class="group-item" data-id="${g.id}" onclick="loadGroupDetails('${g.id}')">
            <strong># ${g.name}</strong>
            <span>${g.members.length} 个成员</span>
        </div>
    `).join('');
}

function renderGroupDetails(group) {
    document.getElementById('currentGroupName').textContent = `# ${group.name}`;
    document.getElementById('memberCount').textContent = `${group.members.length} 成员`;

    memberListEl.innerHTML = group.members.map(m => `
        <div class="member-card" data-member-id="${m.id}">
            <span class="delete-member" onclick="removeMember('${m.id}')">&times;</span>
            <div class="member-name">${m.name}</div>
            <div class="member-controls">
                <div class="control-row">
                    <span>🌡️ 温度</span>
                    <input type="number" class="temp-input" value="${m.temperature}" min="0" max="2" step="0.1" 
                           onchange="updateMemberParam('${m.id}', 'temperature', parseFloat(this.value))">
                </div>
                <div class="control-row">
                    <span>🧠 Thinking</span>
                    <label class="switch">
                        <input type="checkbox" ${m.thinking ? 'checked' : ''} 
                               onchange="updateMemberParam('${m.id}', 'thinking', this.checked)">
                        <span class="slider"></span>
                    </label>
                </div>
            </div>
        </div>
    `).join('');
}

function renderMessages(messages) {
    if (!currentGroup) return;

    // Debug order
    console.log('Rendering messages:', messages.length);
    messages.forEach((m, i) => console.log(`[${i}] ${m.role} (${m.mode || 'free'}): ${m.content.slice(0, 20)}...`));

    let html = '';
    let currentQARow = [];

    messages.forEach(msg => {
        const mode = msg.mode || 'free';

        if (mode === 'qa') {
            if (msg.role === 'user') {
                // Close previous row
                if (currentQARow.length > 0) {
                    html += `<div class="qa-row">${currentQARow.join('')}</div>`;
                    currentQARow = [];
                }
                // Render User QA Message
                html += `
                    <div class="message user" data-mode="${mode}" style="max-width: 100%; margin-top: 20px;">
                        <div class="message-sender">${msg.sender_name || 'User'}</div>
                        <div class="message-content">${formatContent(msg.content)}</div>
                    </div>
                `;
            } else {
                // Assistant QA Message
                currentQARow.push(`
                    <div class="qa-card" data-mode="${mode}">
                        <div class="qa-card-header">
                            <span>${msg.sender_name}</span>
                        </div>
                        <div class="qa-card-content">${formatContent(msg.content)}</div>
                    </div>
                `);
            }
        } else {
            // Free Mode
            // Close previous QA row
            if (currentQARow.length > 0) {
                html += `<div class="qa-row">${currentQARow.join('')}</div>`;
                currentQARow = [];
            }
            // Render Free Message
            html += `
                <div class="message ${msg.role}" data-mode="${mode}">
                    <div class="message-sender">${msg.sender_name || 'User'}</div>
                    <div class="message-content">${formatContent(msg.content)}</div>
                </div>
            `;
        }
    });

    // Close last row
    if (currentQARow.length > 0) {
        html += `<div class="qa-row">${currentQARow.join('')}</div>`;
    }

    messageAreaEl.innerHTML = html;
    scrollToBottom();
}

// renderFreeMessages and renderQAMessages are no longer needed
// function renderFreeMessages...
// function renderQAMessages...

function scrollToBottom() {
    messageAreaEl.scrollTop = messageAreaEl.scrollHeight;
}

// 配置 marked
marked.setOptions({
    highlight: function (code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true
});

function formatContent(text) {
    // 使用 marked 渲染 Markdown
    return marked.parse(text);
}

// ============ Actions ============

async function createGroup() {
    const name = document.getElementById('newGroupName').value;
    const mode = document.getElementById('newGroupMode').value;

    if (!name) return alert('请输入群聊名称');

    try {
        await api('/groups', 'POST', { name, discussion_mode: mode });
        closeModal('createGroupModal');
        loadGroups();
    } catch (e) {
        alert(e.message);
    }
}

async function addMember() {
    if (!currentGroupId) return;

    const model_id = document.getElementById('memberModel').value;
    const description = document.getElementById('memberDescription').value;
    const temperature = parseFloat(document.getElementById('memberTemperature').value);
    const thinking = document.getElementById('memberThinking').checked;

    try {
        await api(`/groups/${currentGroupId}/members`, 'POST', {
            name: model_id,
            model_id,
            description,
            temperature,
            thinking
        });
        closeModal('addMemberModal');
        // 重置表单
        document.getElementById('memberDescription').value = '';
        document.getElementById('memberTemperature').value = '0.7';
        document.getElementById('temperatureValue').textContent = '0.7';
        document.getElementById('memberThinking').checked = false;
        loadGroupDetails(currentGroupId);
    } catch (e) {
        alert(e.message);
    }
}

async function removeMember(memberId) {
    if (!confirm('确定移除该成员吗？')) return;
    try {
        await api(`/groups/${currentGroupId}/members/${memberId}`, 'DELETE');
        loadGroupDetails(currentGroupId);
    } catch (e) {
        alert(e.message);
    }
}

// 编辑成员状态
// State variable editingMemberId moved to top

function openEditMember(memberId, description, temperature, thinking) {
    editingMemberId = memberId;
    document.getElementById('editMemberDescription').value = description;
    document.getElementById('editMemberTemperature').value = temperature;
    document.getElementById('editTemperatureValue').textContent = temperature;
    document.getElementById('editMemberThinking').checked = thinking;
    openModal('editMemberModal');
}

async function updateMember() {
    if (!currentGroupId || !editingMemberId) return;

    const description = document.getElementById('editMemberDescription').value;
    const temperature = parseFloat(document.getElementById('editMemberTemperature').value);
    const thinking = document.getElementById('editMemberThinking').checked;

    try {
        await api(`/groups/${currentGroupId}/members/${editingMemberId}`, 'PATCH', {
            description,
            temperature,
            thinking
        });
        closeModal('editMemberModal');
        loadGroupDetails(currentGroupId);
    } catch (e) {
        alert(e.message);
    }
}

// 实时更新成员单个参数
async function updateMemberParam(memberId, param, value) {
    if (!currentGroupId) return;

    const data = {};
    data[param] = value;

    try {
        await api(`/groups/${currentGroupId}/members/${memberId}`, 'PATCH', data);
    } catch (e) {
        alert('更新失败: ' + e.message);
        loadGroupDetails(currentGroupId); // 恢复原值
    }
}

async function deleteGroup() {
    if (!currentGroupId || !confirm('确定删除该群聊吗？所有记录将丢失。')) return;

    try {
        await api(`/groups/${currentGroupId}`, 'DELETE');
        currentGroupId = null;
        location.reload();
    } catch (e) {
        alert(e.message);
    }
}

async function startDiscussion() {
    if (!currentGroupId || !currentGroup) return;

    const content = document.getElementById('questionInput').value;
    const userName = document.getElementById('userName').value || '用户';
    const maxRounds = parseInt(document.getElementById('maxRounds').value);

    if (!content) return alert('请输入问题或话题');

    // UI Updates before request
    // UI Updates before request
    // 使用界面上的选择覆盖群组默认值 (如果有)
    const selectedMode = document.getElementById('discussionModeSelect').value;
    const isQA = selectedMode === 'qa';
    const mode = selectedMode;
    // Generate unique round ID to avoid DOM ID collisions
    const roundId = Date.now();

    // Render User Message
    const userMsgHtml = `
        <div class="message user" style="${isQA ? 'max-width: 100%; margin-top: 20px;' : ''}">
            <div class="message-sender">${userName}</div>
            <div class="message-content">${formatContent(content)}</div>
        </div>
    `;
    messageAreaEl.insertAdjacentHTML('beforeend', userMsgHtml);

    // [QA Mode] Render Placeholder Cards
    if (isQA) {
        const loadingCards = currentGroup.members.map(m => `
            <div class="qa-card" id="card-${sanitizeId(m.name)}-${roundId}">
                <div class="qa-card-header">
                    <span>${m.name}</span>
                </div>
                <div class="qa-card-content qa-loading">
                    思考中
                </div>
            </div>
        `).join('');
        messageAreaEl.insertAdjacentHTML('beforeend', `<div class="qa-row">${loadingCards}</div>`);
    }

    scrollToBottom();
    document.getElementById('questionInput').value = '';

    // Disable button
    // UI Update: Hide Start, Show Stop
    const btn = document.getElementById('startDiscussionBtn');
    const stopBtn = document.getElementById('stopDiscussionBtn');

    btn.style.display = 'none';
    stopBtn.style.display = 'block';

    // Init AbortController
    if (abortController) abortController.abort();
    abortController = new AbortController();

    // QA Mode buffer for each member
    const qaBuffers = {};
    if (isQA) {
        currentGroup.members.forEach(m => qaBuffers[m.name] = '');
    }

    try {
        const response = await fetch(`${API_BASE}/groups/${currentGroupId}/discuss/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, user_name: userName, max_rounds: maxRounds, mode }),
            signal: abortController.signal
        });

        if (!response.ok) throw new Error('讨论请求失败');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));

                    if (data.type === 'message') {
                        if (isQA) {
                            // QA Mode Update
                            const cardId = `card-${sanitizeId(data.sender_name)}-${roundId}`;
                            const cardContentEl = document.querySelector(`#${cardId} .qa-card-content`);

                            if (cardContentEl) {
                                // For simplicity, we just replace content as our backend yields full content per agent completion
                                // But wait, group_chat.py yields {"sender": ..., "content": ...}
                                // stream_qa_discussion yields ONCE per agent when done.
                                // So we can just set innerHTML.
                                cardContentEl.classList.remove('qa-loading');
                                cardContentEl.innerHTML = formatContent(data.content);
                            }
                        } else {
                            // Free Mode Update (Append message)
                            messageAreaEl.insertAdjacentHTML('beforeend', `
                                <div class="message assistant">
                                    <div class="message-sender">${data.sender_name}</div>
                                    <div class="message-content">${formatContent(data.content)}</div>
                                </div>
                            `);
                        }
                        scrollToBottom();
                    } else if (data.type === 'error') {
                        alert('讨论出错: ' + data.message);
                    }
                }
            }
        }

    } catch (e) {
        if (e.name === 'AbortError') {
            messageAreaEl.insertAdjacentHTML('beforeend', `
                <div class="message system" style="text-align: center; color: #ef4444; margin: 10px 0;">
                    🛑 讨论已终止
                </div>
            `);
        } else {
            alert('讨论出错: ' + e.message);
            console.error(e);
        }
    } finally {
        btn.style.display = 'block';
        btn.disabled = false;
        btn.textContent = '🚀 开始讨论';

        const stopBtn = document.getElementById('stopDiscussionBtn');
        if (stopBtn) stopBtn.style.display = 'none';
        abortController = null;

        scrollToBottom();
        // 刷新上下文状态
        await loadContextStats(currentGroupId);
    }
}

function sanitizeId(str) {
    // Generate safe CSS ID
    return str.replace(/[^a-zA-Z0-9-_]/g, '_');
}

async function summarizeDiscussion() {
    console.log('Summarize discussion triggered');
    if (!currentGroupId) {
        console.error('No current group ID');
        return;
    }

    const content = document.getElementById('questionInput').value.trim();
    const btn = document.getElementById('summarizeBtn');

    btn.disabled = true;
    btn.textContent = '📝 总结中...';

    // Optional: Show user instruction if provided
    if (content) {
        const userName = document.getElementById('userNameInput').value || 'User';
        messageAreaEl.insertAdjacentHTML('beforeend', `
            <div class="message user">
                <div class="message-sender">${userName}</div>
                <div class="message-content">${formatContent(content)}</div>
            </div>
        `);
        document.getElementById('questionInput').value = '';
        scrollToBottom();
    }

    // Show placeholder
    const summaryId = `summary-${Date.now()}`;
    messageAreaEl.insertAdjacentHTML('beforeend', `
        <div class="message assistant" id="${summaryId}" style="border-left: 4px solid #5a6b7c; background-color: #f8f9fa;">
            <div class="message-sender">📝 总结助手</div>
            <div class="message-content">
                <div style="color: #666; font-style: italic;">
                    正在分析对话历史并生成总结... ⏳
                </div>
            </div>
        </div>
    `);
    scrollToBottom();

    try {
        const payload = content ? { instruction: content } : {};

        const response = await fetch(`${API_BASE}/groups/${currentGroupId}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('总结请求失败');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));

                    if (data.type === 'message') {
                        // Update placeholder with actual content
                        const card = document.getElementById(summaryId);
                        if (card) {
                            card.querySelector('.message-sender').textContent = `📝 ${data.sender_name}`;
                            card.querySelector('.message-content').innerHTML = formatContent(data.content);
                        } else {
                            // Fallback if placeholder missing
                            messageAreaEl.insertAdjacentHTML('beforeend', `
                                <div class="message assistant" style="border-left: 4px solid #5a6b7c; background-color: #f8f9fa;">
                                    <div class="message-sender">📝 ${data.sender_name}</div>
                                    <div class="message-content">${formatContent(data.content)}</div>
                                </div>
                            `);
                        }
                        scrollToBottom();
                    } else if (data.type === 'error') {
                        const card = document.getElementById(summaryId);
                        if (card) {
                            card.querySelector('.message-content').innerHTML = `<span style="color:red">Error: ${data.message}</span>`;
                        } else {
                            alert('总结出错: ' + data.message);
                        }
                    }
                }
            }
        }

    } catch (e) {
        alert('总结出错: ' + e.message);
        console.error(e);
    } finally {
        btn.disabled = false;
        btn.textContent = '📝 得出结论';
        scrollToBottom();
        // 刷新上下文状态
        await loadContextStats(currentGroupId);
    }
}

// ============ Event Listeners ============

function setupEventListeners() {
    // Buttons - 使用 addEventListener 确保不会被覆盖
    document.getElementById('createGroupBtn').addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        openModal('createGroupModal');
    });

    document.getElementById('addMemberBtn').addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        openModal('addMemberModal');
    });

    document.getElementById('deleteGroupBtn').addEventListener('click', (e) => {
        e.preventDefault();
        deleteGroup();
    });

    document.getElementById('startDiscussionBtn').addEventListener('click', (e) => {
        e.preventDefault();
        startDiscussion();
    });

    const stopBtn = document.getElementById('stopDiscussionBtn');
    if (stopBtn) {
        stopBtn.addEventListener('click', (e) => {
            e.preventDefault();
            if (abortController) {
                console.log('Sending abort signal...');
                abortController.abort();
            }
        });
    }

    const summarizeBtn = document.getElementById('summarizeBtn');
    if (summarizeBtn) {
        summarizeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            summarizeDiscussion();
        });
    }

    document.getElementById('confirmCreateGroup').addEventListener('click', (e) => {
        e.preventDefault();
        createGroup();
    });

    document.getElementById('confirmAddMember').addEventListener('click', (e) => {
        e.preventDefault();
        addMember();
    });

    // 管理员设置
    document.getElementById('setManagerBtn').addEventListener('click', (e) => {
        e.preventDefault();
        openModal('setManagerModal');
    });

    document.getElementById('confirmSetManager').addEventListener('click', async (e) => {
        e.preventDefault();
        const model_id = document.getElementById('managerModel').value;
        const temperature = parseFloat(document.getElementById('managerTemperature').value);
        const thinking = document.getElementById('managerThinking').checked;

        try {
            await api(`/groups/${currentGroupId}/manager`, 'PUT', {
                model_id,
                temperature,
                thinking
            });
            closeModal('setManagerModal');
            alert('管理员配置设置成功');
        } catch (e) {
            alert(e.message);
        }
    });

    // 管理员温度滑块同步
    document.getElementById('managerTemperature').addEventListener('input', (e) => {
        document.getElementById('managerTemperatureValue').textContent = e.target.value;
    });

    // 温度滑块同步
    document.getElementById('memberTemperature').addEventListener('input', (e) => {
        document.getElementById('temperatureValue').textContent = e.target.value;
    });

    // 编辑成员温度滑块同步
    document.getElementById('editMemberTemperature').addEventListener('input', (e) => {
        document.getElementById('editTemperatureValue').textContent = e.target.value;
    });

    // 编辑成员确认按钮
    document.getElementById('confirmEditMember').addEventListener('click', (e) => {
        e.preventDefault();
        updateMember();
    });

    // 模态框关闭按钮 - 只处理 × 按钮
    document.querySelectorAll('.modal-header .close-btn').forEach(btn => {
        btn.addEventListener('click', function (e) {
            e.preventDefault();
            this.closest('.modal').style.display = 'none';
        });
    });

    // 模态框取消按钮
    document.querySelectorAll('.modal-footer .cancel-btn').forEach(btn => {
        btn.addEventListener('click', function (e) {
            e.preventDefault();
            this.closest('.modal').style.display = 'none';
        });
    });

    // 点击模态框背景关闭
    window.addEventListener('click', function (event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
        }
    });

    // 刷新上下文状态按钮
    document.getElementById('refreshContextBtn').addEventListener('click', async (e) => {
        e.preventDefault();
        if (currentGroupId) {
            await loadContextStats(currentGroupId);
        }
    });

    const thresholdSlider = document.getElementById('thresholdSlider');
    if (thresholdSlider) {
        const thresholdVal = document.getElementById('thresholdValue');

        thresholdSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (thresholdVal) thresholdVal.textContent = `${(val * 100).toFixed(0)}%`;
        });

        thresholdSlider.addEventListener('change', async (e) => {
            const val = parseFloat(e.target.value);
            if (currentGroupId) {
                try {
                    await api(`/groups/${currentGroupId}/compression/threshold`, 'PUT', { threshold: val });
                    // Refresh stats to update marker position
                    await loadContextStats(currentGroupId);
                } catch (err) {
                    console.error(err);
                    alert('更新阈值失败: ' + err.message);
                }
            }
        });
    }
}

function openModal(id) {
    document.getElementById(id).style.display = 'block';
}

function closeModal(id) {
    document.getElementById(id).style.display = 'none';
}

// ============ Context Stats ============

async function loadContextStats(groupId) {
    try {
        const stats = await api(`/groups/${groupId}/context/stats`);
        renderContextStats(stats);
    } catch (e) {
        console.error('加载上下文状态失败:', e);
    }
}

function renderContextStats(stats) {
    const currentTokens = stats.current_tokens || 0;

    // Get max tokens from new structure or fallback
    let maxTokens = 128000;
    if (stats.dynamic_context_window && stats.dynamic_context_window.min_context_window) {
        maxTokens = stats.dynamic_context_window.min_context_window;
    } else if (stats.compression_config && stats.compression_config.max_tokens) {
        maxTokens = stats.compression_config.max_tokens;
    } else if (stats.max_tokens) {
        maxTokens = stats.max_tokens;
    }

    // Calculate ratio
    const usageRatio = maxTokens > 0 ? currentTokens / maxTokens : 0;
    const usagePercent = (usageRatio * 100).toFixed(1);

    // Get threshold settings for tooltip and logic
    const thresholdRatio = (stats.compression_config && stats.compression_config.threshold_ratio) || 0.8;
    const thresholdTokens = (stats.compression_config && stats.compression_config.threshold_tokens)
        || Math.floor(maxTokens * thresholdRatio);

    // Update new compact UI
    const currentTokensEl = document.getElementById('currentTokens');
    if (currentTokensEl) currentTokensEl.textContent = formatTokens(currentTokens);

    const maxTokensEl = document.getElementById('maxTokens');
    if (maxTokensEl) maxTokensEl.textContent = formatTokens(maxTokens);

    const progressTextEl = document.getElementById('contextProgressText');
    if (progressTextEl) progressTextEl.textContent = `${usagePercent}%`;

    // Update threshold marker position (percentage)
    const markerEl = document.getElementById('thresholdMarker');
    if (markerEl) {
        markerEl.style.left = `${thresholdRatio * 100}%`;
        const thresholdPercent = (thresholdRatio * 100).toFixed(0);
        markerEl.parentElement.title = `压缩阈值: ${thresholdPercent}% (${formatTokens(thresholdTokens)})`;
    }

    // Update progress bar
    const progressBar = document.getElementById('contextProgressBar');
    progressBar.style.width = `${Math.min(usagePercent, 100)}%`;

    // Color coding based on usage
    progressBar.classList.remove('warning', 'danger');
    if (usageRatio >= thresholdRatio) {
        progressBar.classList.add('danger');
    } else if (usageRatio >= thresholdRatio * 0.8) {
        progressBar.classList.add('warning');
    }

    // Show stats container if hidden
    const statsContainer = document.getElementById('minContextStats');
    if (statsContainer) statsContainer.style.display = 'flex';
}

function formatTokens(tokens) {
    if (tokens >= 1000000) {
        return (tokens / 1000000).toFixed(1) + 'M';
    } else if (tokens >= 1000) {
        return (tokens / 1000).toFixed(1) + 'K';
    }
    return tokens.toString();
}

// Start
init();
