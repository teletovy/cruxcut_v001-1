// public/sw.js
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'START_POLLING') {
    const { jobId, apiUrl } = event.data;
    startPolling(jobId, apiUrl);
  }
});

let pollingInterval;

async function startPolling(jobId, apiUrl) {
  // 기존 폴링이 있다면 중단
  if (pollingInterval) {
    clearInterval(pollingInterval);
  }

  pollingInterval = setInterval(async () => {
    try {
      const response = await fetch(`${apiUrl}/status/${jobId}`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const status = await response.json();
      
      // 메인 스레드로 상태 전송
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'JOB_STATUS_UPDATE',
            jobId,
            status
          });
        });
      });
      
      // 작업 완료 시 폴링 중단
      if (status.status === 'completed' || status.status === 'failed') {
        clearInterval(pollingInterval);
        
        // 완료 알림 표시 (에러 핸들링 추가)
        try {
          await self.registration.showNotification('CruxCut 처리 완료', {
            body: status.status === 'completed' 
              ? '영상 처리가 완료되었습니다!' 
              : '영상 처리 중 오류가 발생했습니다.',
            icon: '/icon-192x192.svg',
            badge: '/badge-72x72.png',
            tag: 'video-processing'
          });
        } catch (error) {
          console.log('알림 표시 실패:', error);
        }
      }
    } catch (error) {
      console.error('Polling error:', error);
      
      // 에러 발생 시 메인 스레드에 알림
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'POLLING_ERROR',
            jobId,
            error: error.message
          });
        });
      });
    }
  }, 3000); // 3초마다 확인
}

// 알림 클릭 시 앱으로 이동
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  event.waitUntil(
    clients.matchAll().then(clientList => {
      // 이미 열린 탭이 있으면 focus
      for (const client of clientList) {
        if (client.url === '/' && 'focus' in client) {
          return client.focus();
        }
      }
      // 없으면 새 탭 열기
      if (clients.openWindow) {
        return clients.openWindow('/');
      }
    })
  );
});