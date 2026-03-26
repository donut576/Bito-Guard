import React, { useState, useMemo } from 'react';
import { useApi } from '../hooks/useApi';
import { ENDPOINTS } from '../api/endpoints';
import GlowCard from './shared/GlowCard';
import ApiPill from './shared/ApiPill';

const FraudExplanationPanel = () => {
  const fraudData = useApi(ENDPOINTS.fraud_explanation(10));
  const [selectedUserId, setSelectedUserId] = useState(null);

  const users = fraudData.data?.explanations || [];
  const selectedUser = useMemo(() => {
    if (!users || users.length === 0) return null;
    if (!selectedUserId) return users[0];
    return users.find(user => user.user_id === selectedUserId) || users[0];
  }, [users, selectedUserId]);

  const formatFeatures = (features) => {
    if (!features || features === '') return [];
    return features.split(' / ').filter(f => f.trim() !== '');
  };

  const getRiskColor = (prob) => {
    if (prob >= 0.8) return '#ff3366';
    if (prob >= 0.6) return '#ff9933';
    return '#ffcc33';
  };

  return (
    <div className="section">
      <div className="section-title">詐騙用戶分析</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <ApiPill loading={fraudData.loading} error={fraudData.error} isMock={fraudData.isMock} />
        {fraudData.data && (
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            共 {fraudData.data.total} 位高風險用戶
          </span>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* 誰是詐騙 - Fraud Users List */}
        <GlowCard>
          <div style={{ marginBottom: 16 }}>
            <h3 style={{ color: 'var(--neon-red)', margin: 0, fontSize: 16, fontWeight: 600 }}>
              🚨 誰是詐騙
            </h3>
            <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0', fontSize: 12 }}>
              根據模型預測的最高風險用戶
            </p>
          </div>

          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            {users.length > 0 ? users.map((user, index) => {
              const active = user.user_id === selectedUser?.user_id;
              return (
                <div
                  key={user.user_id}
                  onClick={() => setSelectedUserId(user.user_id)}
                  style={{
                    cursor: 'pointer',
                    padding: '12px',
                    marginBottom: 8,
                    borderRadius: 8,
                    background: active ? 'rgba(255, 51, 102, 0.2)' : 'rgba(255, 51, 102, 0.05)',
                    border: active ? '1px solid rgba(255, 51, 102, 0.35)' : '1px solid rgba(255, 51, 102, 0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                  }}
                >
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: active ? 'var(--neon-cyan)' : 'var(--text-primary)' }}>
                      用戶 {user.user_id}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                      風險等級: <span style={{
                        color: getRiskColor(user.fraud_prob),
                        fontWeight: 600
                      }}>
                        {user.fraud_prob_percent}%
                      </span>
                    </div>
                  </div>
                  <div style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: getRiskColor(user.fraud_prob)
                  }} />
                </div>
              );
            }) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20 }}>
                載入中...
              </div>
            )}
          </div>
        </GlowCard>

        {/* 詐騙原因 - Fraud Reasons */}
        <GlowCard>
          <div style={{ marginBottom: 16 }}>
            <h3 style={{ color: 'var(--neon-orange)', margin: 0, fontSize: 16, fontWeight: 600 }}>
              🔍 詐騙原因
            </h3>
            <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0', fontSize: 12 }}>
              SHAP解釋的關鍵風險特徵
            </p>
          </div>

          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            {selectedUser ? (
              <div>
                <div style={{
                  padding: '12px',
                  marginBottom: 12,
                  borderRadius: 8,
                  background: 'rgba(255, 153, 51, 0.05)',
                  border: '1px solid rgba(255, 153, 51, 0.2)'
                }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
                    用戶 {selectedUser.user_id} 的風險特徵
                  </div>

                  {/* KYC Features */}
                  {formatFeatures(selectedUser.kyc_features).length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--neon-cyan)', marginBottom: 4 }}>
                        👤 用戶基本資訊
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        {formatFeatures(selectedUser.kyc_features).map((feature, i) => (
                          <div key={i} style={{ marginBottom: 2 }}>• {feature}</div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* TWD Features */}
                  {formatFeatures(selectedUser.twd_features).length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--neon-green)', marginBottom: 4 }}>
                        💰 台幣出入金
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        {formatFeatures(selectedUser.twd_features).map((feature, i) => (
                          <div key={i} style={{ marginBottom: 2 }}>• {feature}</div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Crypto Features */}
                  {formatFeatures(selectedUser.crypto_features).length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--neon-purple)', marginBottom: 4 }}>
                        ₿ 虛擬貨幣轉帳
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        {formatFeatures(selectedUser.crypto_features).map((feature, i) => (
                          <div key={i} style={{ marginBottom: 2 }}>• {feature}</div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Wallet Features */}
                  {formatFeatures(selectedUser.wallet_features).length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--neon-yellow)', marginBottom: 4 }}>
                        👛 錢包風險
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        {formatFeatures(selectedUser.wallet_features).map((feature, i) => (
                          <div key={i} style={{ marginBottom: 2 }}>• {feature}</div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Cross Features */}
                  {formatFeatures(selectedUser.cross_features).length > 0 && (
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--neon-pink)', marginBottom: 4 }}>
                        🔗 綜合風險指標
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        {formatFeatures(selectedUser.cross_features).map((feature, i) => (
                          <div key={i} style={{ marginBottom: 2 }}>• {feature}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 11 }}>
                  點擊上方用戶查看不同人的風險特徵
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20 }}>
                載入中...
              </div>
            )}
          </div>
        </GlowCard>
      </div>
    </div>
  );
};

export default FraudExplanationPanel;