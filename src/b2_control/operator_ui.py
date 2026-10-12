"""Loopback-only browser operator panel for B2 policy velocity commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import threading
from typing import Any
import webbrowser

import numpy as np

from .interactive_command import InteractiveVelocityCommand
from .timed_test import TEST_DEFINITIONS


@dataclass
class OperatorStatus:
    phase: str = "WAITING LOWSTATE"
    gateway_connected: bool = False
    state_sequence: int = 0
    inference_ms: float = 0.0
    tracking_error_rad: float = 0.0
    last_event: str = "ZERO/HOLD"
    control_mode: str = "SIMULATION"
    mode_detail: str = "仿真/外部 gateway 模式"
    mode_switch_enabled: bool = False
    mode_transition: bool = False
    motion_commands_enabled: bool = True
    input_source: str = "WEB KEYBOARD"
    input_source_switch_enabled: bool = False
    lifecycle_state: str = "DISCONNECTED"
    output_mode: str = "NONE"


TEST_TEMPLATES: dict[str, dict[str, Any]] = {
    test_id: {
        "label": definition.name,
        "kind": definition.kind,
        "duration_s": definition.duration_s,
        "vx": (
            "selected"
            if test_id == "forward-2s"
            else definition.default_speed_m_s
        ),
        "scene": "stairs" if test_id == "stairs" else "flat",
    }
    for test_id, definition in TEST_DEFINITIONS.items()
}


_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unitree B2 操作台</title>
<style>
  *{box-sizing:border-box} body{margin:0;background:#0b1016;color:#edf4fa;font-family:Inter,"Noto Sans SC",system-ui,sans-serif;min-height:100vh}
  .shell{width:min(1120px,calc(100% - 32px));margin:24px auto;display:grid;gap:16px}
  header{display:flex;justify-content:space-between;align-items:center;padding:18px 20px;background:#121a23;border:1px solid #263441;border-radius:14px}
  h1{font-size:22px;margin:0;letter-spacing:.02em}.sub{color:#8295a6;font-size:12px;margin-top:4px}
  .badge{padding:8px 12px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.08em;background:#341b21;color:#ff94a3;border:1px solid #74313c}
  .badge.live{background:#102d25;color:#7df1be;border-color:#24634f}
  main{display:grid;grid-template-columns:1.05fr .95fr;gap:16px}.panel{background:#121a23;border:1px solid #263441;border-radius:14px;padding:18px}
  .panel-title{color:#91a5b6;font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px}
  .velocity{display:grid;grid-template-columns:2fr 1fr 1fr;gap:10px}.metric{background:#0d141c;border:1px solid #243340;border-radius:10px;padding:14px}
  .metric strong{display:block;font-size:36px;line-height:1.1;font-variant-numeric:tabular-nums}.metric.small strong{font-size:24px}.metric span{font-size:11px;color:#7890a2}
  canvas{width:100%;height:226px;background:#0d141c;border:1px solid #243340;border-radius:10px}
  .statusline{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px;color:#9cb0c0;font-size:12px}.statusline b{color:#eaf2f8;font-variant-numeric:tabular-nums}
  .movement{display:grid;grid-template-columns:1fr 1.4fr 1fr;gap:8px}.move{border:1px solid #2e4353;background:#17232d;color:#dbe7ef;border-radius:9px;padding:15px 12px;cursor:pointer;font-weight:700;touch-action:none;user-select:none}.move:hover,.move.held{background:#205069;border-color:#49c9e9}.stop{grid-column:1/-1;background:#a62f3e;border:1px solid #e26070;color:white;border-radius:11px;padding:16px;font-weight:900;font-size:16px;cursor:pointer}.stop:hover{background:#c23a4b}
  .speed-presets{display:grid;grid-template-columns:repeat(9,1fr);gap:8px}.speed-button{border:1px solid #365064;background:#172630;color:#dbe7ef;border-radius:9px;padding:12px 6px;cursor:pointer;font-weight:900}.speed-button small{display:block;color:#8298a8;font-weight:500;margin-top:3px}.speed-button.active{background:#205069;border-color:#49c9e9;outline:2px solid #49c9e9}.speed-button:disabled{opacity:.35;cursor:not-allowed}
  .event{margin-top:12px;color:#7df1be;font-size:13px;min-height:20px}.keys{color:#8397a8;font-size:12px;line-height:1.8}.keys kbd{background:#202c36;border:1px solid #3a4a57;border-radius:5px;padding:2px 6px;color:#eaf2f8}
  .mode-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.mode-button{border:1px solid #365064;border-radius:11px;padding:15px;font-size:15px;font-weight:900;cursor:pointer;color:#eaf2f8;background:#172630}.mode-button.native{border-color:#50754e;background:#18321f}.mode-button.custom{border-color:#35728b;background:#123747}.mode-button.active{outline:3px solid #65d7f2;outline-offset:1px}.mode-button:disabled,.move:disabled,.stop:disabled{opacity:.35;cursor:not-allowed}.mode-detail{margin-top:11px;color:#9eb3c3;font-size:13px}.warning{margin-top:9px;color:#ffbd72;font-size:12px}
  .telemetry-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.telemetry-grid .metric strong{font-size:20px}.joint-wrap{overflow:auto;max-height:330px;margin-top:12px;border:1px solid #243340;border-radius:9px}table{width:100%;border-collapse:collapse;font:12px ui-monospace,monospace}th,td{padding:7px 9px;text-align:right;border-bottom:1px solid #22313d;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#17232d;color:#8fa5b6}.fault-text{color:#ff8f9f;font-weight:900}
  .lifecycle{display:flex;gap:6px;overflow:auto;padding-bottom:3px}.life{flex:0 0 auto;border:1px solid #304251;border-radius:999px;padding:7px 10px;color:#748a9b;font-size:11px;font-weight:800}.life.active{color:#081016;background:#65d7f2;border-color:#65d7f2}.state-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}.state-grid strong{font-size:21px}.ops-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.ops-grid .metric strong{font-size:18px}.fault-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.fault-item{background:#0d141c;border:1px solid #382832;border-radius:9px;padding:11px}.fault-item span{display:block;color:#8195a6;font-size:10px}.fault-item strong{display:block;margin-top:5px;font:700 14px ui-monospace,monospace;overflow-wrap:anywhere}.test-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.test-button{border:1px solid #365064;background:#172630;color:#dbe7ef;border-radius:9px;padding:11px 7px;cursor:pointer;font-weight:800}.test-button:disabled{opacity:.35;cursor:not-allowed}
  @media(max-width:760px){main{grid-template-columns:1fr}.velocity{grid-template-columns:1fr 1fr}.velocity .metric:first-child{grid-column:1/-1}.speed-presets{grid-template-columns:repeat(3,1fr)}.telemetry-grid,.ops-grid,.fault-grid,.test-grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="shell">
  <header><div><h1>Unitree B2 Policy 操作台</h1><div class="sub">45维 observation · ONNX 50 Hz · DDS LowCmd 500 Hz</div></div><div id="connection" class="badge">等待 LOWSTATE</div></header>
  <main>
    <section class="panel">
      <div class="panel-title">当前速度命令</div>
      <div class="velocity">
        <div class="metric"><strong id="vx">0.00</strong><span>vx · 前进 m/s</span></div>
        <div class="metric small"><strong id="vy">0.00</strong><span>vy · 横移 m/s</span></div>
        <div class="metric small"><strong id="wz">0.00</strong><span>wz · 转向 rad/s</span></div>
      </div>
      <div class="statusline"><div>状态<br><b id="phase">WAITING</b></div><div>推理<br><b id="infer">0.00 ms</b></div><div>最大跟踪误差<br><b id="error">0.000 rad</b></div></div><div class="sub" style="margin-top:10px">命令源：<b id="inputSource">WEB KEYBOARD</b></div>
      <div id="event" class="event">ZERO/HOLD</div>
    </section>
    <section class="panel"><div class="panel-title">运动方向</div><canvas id="motion" width="500" height="226"></canvas><div class="statusline"><div>LowState 序号<br><b id="sequence">0</b></div><div>W/S 当前档位<br><b id="forwardSpeed">-- m/s</b></div><div>A/D 固定速度<br><b id="lateralSpeed">-- m/s</b></div></div></section>
  </main>
  <section class="panel">
    <div class="panel-title">接管生命周期与当前真实输出</div>
    <div class="lifecycle"><span class="life" data-life="DISCONNECTED">断开</span><span class="life" data-life="READ_ONLY_CHECK">只读检查</span><span class="life" data-life="NATIVE_CONFIRMED">原生确认</span><span class="life" data-life="ARMED">已武装</span><span class="life" data-life="CUSTOM_INTERPOLATION">自研插值</span><span class="life" data-life="HOLD">HOLD</span><span class="life" data-life="WALK">WALK</span><span class="life" data-life="STOPPING">STOPPING</span><span class="life" data-life="NATIVE">NATIVE</span><span class="life" data-life="FAULT">FAULT</span></div>
    <div class="state-grid"><div class="metric small"><strong id="lifecycleState">DISCONNECTED</strong><span>权威生命周期状态</span></div><div class="metric small"><strong id="outputMode">NONE</strong><span>当前底层实际输出</span></div></div>
  </section>
  <section class="panel"><div class="panel-title">真机遥测与保护状态</div><div class="telemetry-grid"><div class="metric small"><strong id="safetyFlag">NORMAL</strong><span>安全锁存</span></div><div class="metric small"><strong id="batterySoc">--%</strong><span>电池 SOC</span></div><div class="metric small"><strong id="batteryCurrent">--</strong><span>BMS 电流 raw</span></div><div class="metric small"><strong id="motorTemp">--°C</strong><span>最高电机温度</span></div><div class="metric small"><strong id="maxTau">--</strong><span>max |tau_est|</span></div><div class="metric small"><strong id="lostSum">--</strong><span>motor lost 累计</span></div></div><div class="joint-wrap"><table><thead><tr><th>关节</th><th>target</th><th>actual</th><th>error</th><th>dq</th><th>tau_est</th><th>温度</th><th>lost</th></tr></thead><tbody id="jointRows"><tr><td colspan="8">等待 LowState</td></tr></tbody></table></div></section>
  <section class="panel"><div class="panel-title">控制与通信性能</div><div class="ops-grid"><div class="metric small"><strong id="crcErrors">--</strong><span>LowState CRC 错误</span></div><div class="metric small"><strong id="ddsHz">-- Hz</strong><span>DDS LowState 实际频率</span></div><div class="metric small"><strong id="jitterNow">-- ms</strong><span>500 Hz 当前抖动</span></div><div class="metric small"><strong id="jitterMax">-- ms</strong><span>500 Hz 最大抖动</span></div><div class="metric small"><strong id="policyE2e">-- ms</strong><span>Policy 端到端延迟</span></div><div class="metric small"><strong id="cpuPercent">--%</strong><span>控制进程 CPU</span></div><div class="metric small"><strong id="packetLoss">--%</strong><span>网络丢包</span></div><div class="metric small"><strong id="inferDetail">-- ms</strong><span>ONNX 单次推理</span></div></div></section>
  <section class="panel"><div class="panel-title">故障原因</div><div class="fault-grid"><div class="fault-item"><span>关节</span><strong id="faultJoint">--</strong></div><div class="fault-item"><span>指标</span><strong id="faultMetric">--</strong></div><div class="fault-item"><span>触发值</span><strong id="faultValue">--</strong></div><div class="fault-item"><span>阈值</span><strong id="faultThreshold">--</strong></div><div class="fault-item"><span>持续时间</span><strong id="faultDuration">--</strong></div><div class="fault-item"><span>触发后输出</span><strong id="faultOutput">--</strong></div></div></section>
  <section class="panel"><div class="panel-title">限时测试模板 · 确认后交给控制循环执行</div><div class="test-grid"><button class="test-button" data-test="stand-10s" data-motion>站立 10 秒</button><button class="test-button" data-test="forward-2s" data-motion>前进 2 秒</button><button class="test-button" data-test="speed-0.1" data-motion>0.1 m/s · 2 秒</button><button class="test-button" data-test="speed-0.25" data-motion>0.25 m/s · 2 秒</button><button class="test-button" data-test="speed-0.4" data-motion>0.4 m/s · 2 秒</button><button class="test-button" data-test="speed-0.6" data-motion>0.6 m/s · 2 秒</button><button class="test-button" data-test="stairs" data-motion>楼梯测试</button><button class="test-button" id="cancelTest">立即取消测试并归零</button></div><div class="warning">模板不会绕过控制状态机；仅在运动命令已启用时接受，并要求再次确认。取消测试始终可用。</div></section>
  <section class="panel">
    <div class="panel-title">W/S 速度档位 · 只选择速度，不会触发运动</div>
    <div class="speed-presets"><button class="speed-button" data-speed="1" data-motion>1<small>0.1 m/s</small></button><button class="speed-button" data-speed="2" data-motion>2<small>0.2 m/s</small></button><button class="speed-button" data-speed="3" data-motion>3<small>0.3 m/s</small></button><button class="speed-button" data-speed="4" data-motion>4<small>0.4 m/s</small></button><button class="speed-button" data-speed="5" data-motion>5<small>0.5 m/s</small></button><button class="speed-button" data-speed="6" data-motion>6<small>0.6 m/s</small></button><button class="speed-button" data-speed="7" data-motion>7<small>0.7 m/s</small></button><button class="speed-button" data-speed="8" data-motion>8<small>0.8 m/s</small></button><button class="speed-button" data-speed="9" data-motion>9<small>0.9 m/s</small></button></div>
  </section>
  <section class="panel">
    <div class="panel-title">Locomotion 控制权</div>
    <div class="mode-row"><button id="nativeMode" class="mode-button native">原生 locomotion</button><button id="customMode" class="mode-button custom">自研 locomotion</button></div>
    <div id="modeDetail" class="mode-detail">正在读取控制模式…</div>
    <div class="warning">切换会先强制速度归零。自研→原生还需关节连续静止 0.5 秒；门控失败时不会强制切换。</div>
  </section>
  <section class="panel"><div class="panel-title">运动命令源</div><div class="mode-row"><button id="keyboardInput" class="mode-button">键盘控制</button><button id="remoteInput" class="mode-button">手持遥控器</button></div><div class="warning">仅在原生 locomotion 状态允许切换；切换会先清零全部速度。遥控器保持 A=policy、R2=learned hold、L2+B=锁存急停。</div></section>
  <section class="panel">
    <div class="panel-title">按住平移 · 松开停止</div>
    <div class="movement"><button class="move" data-hold="a" data-motion>A 按住左移</button><button class="move" data-hold="w" data-motion>W 按住前进</button><button class="move" data-hold="d" data-motion>D 按住右移</button><button class="move" data-step="q" data-motion>Q 角速度 +<span class="yaw-step">0.10</span></button><button class="move" data-hold="s" data-motion>S 按住后退</button><button class="move" data-step="e" data-motion>E 角速度 -<span class="yaw-step">0.10</span></button><button class="stop" data-stop data-motion>停止运动 / LEARNED HOLD（Space / X / Esc）</button></div>
  </section>
  <section class="panel keys"><kbd>1–9</kbd> 选择 W/S 速度 0.1–0.9 m/s（不会自行运动）　<kbd>W/S</kbd> 按住前进/后退，松开停止　<kbd>A/D</kbd> 按住横移，松开停止　<kbd>Q/E</kbd> 每次改变角速度 ±<span id="yawStep">0.10</span> rad/s　<kbd>Space/X/Esc</kbd> 全部清零</section>
</div>
<script>
const token="__TOKEN__", api=`/api/${token}`, $=id=>document.getElementById(id); let last={vx:0,vy:0,wz:0},profile={forward_speed:.9,max_forward_speed:.9,lateral_speed:.25,yaw_step:.1};const held=new Set();
async function sendKey(key,action='press'){try{const r=await fetch(`${api}/key`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key,action})});if(!r.ok)throw Error(await r.text());await refresh()}catch(e){$('event').textContent=`命令失败: ${e.message}`}}
function pressHeld(key,button=null){if(held.has(key))return;held.add(key);if(button)button.classList.add('held');sendKey(key,'press')}
function releaseHeld(key,button=null){if(!held.delete(key))return;if(button)button.classList.remove('held');sendKey(key,'release')}
function releaseAll(){for(const key of [...held])releaseHeld(key);}
async function switchMode(mode){const label=mode==='native'?'原生 locomotion':'自研 locomotion';if(!confirm(`确认切换到${label}？\n系统会先将速度命令清零。`))return;held.clear();document.querySelectorAll('.move.held').forEach(b=>b.classList.remove('held'));try{const r=await fetch(`${api}/mode`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,confirmed:true})});if(!r.ok)throw Error(await r.text());await refresh()}catch(e){$('event').textContent=`切换失败: ${e.message}`}}
$('nativeMode').addEventListener('click',()=>switchMode('native'));$('customMode').addEventListener('click',()=>switchMode('custom'));
async function switchSource(source){const label=source==='keyboard'?'键盘':'手持遥控器';if(!confirm(`确认切换到${label}控制？\n必须先处于原生 locomotion，系统会清零速度。`))return;releaseAll();try{const r=await fetch(`${api}/source`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source,confirmed:true})});if(!r.ok)throw Error(await r.text());await refresh()}catch(e){$('event').textContent=`命令源切换失败: ${e.message}`}}$('keyboardInput').addEventListener('click',()=>switchSource('keyboard'));$('remoteInput').addEventListener('click',()=>switchSource('remote'));
document.querySelectorAll('[data-hold]').forEach(b=>{const k=b.dataset.hold;b.addEventListener('pointerdown',e=>{if(b.disabled)return;e.preventDefault();b.setPointerCapture(e.pointerId);pressHeld(k,b)});const release=e=>{e.preventDefault();releaseHeld(k,b)};b.addEventListener('pointerup',release);b.addEventListener('pointercancel',release);b.addEventListener('lostpointercapture',()=>releaseHeld(k,b))});
document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>sendKey(b.dataset.step,'press')));document.querySelector('[data-stop]').addEventListener('click',()=>{held.clear();document.querySelectorAll('.move.held').forEach(b=>b.classList.remove('held'));sendKey('x','press')});
document.querySelectorAll('[data-speed]').forEach(b=>b.addEventListener('click',()=>sendKey(b.dataset.speed,'press')));
async function requestTest(template){if(!confirm(`确认启动限时测试 ${template}？\n控制循环将自动归零并按模板执行。`))return;releaseAll();try{const r=await fetch(`${api}/test`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({template,confirmed:true})});if(!r.ok)throw Error(await r.text());await refresh()}catch(e){$('event').textContent=`测试请求失败: ${e.message}`}}
document.querySelectorAll('[data-test]').forEach(b=>b.addEventListener('click',()=>requestTest(b.dataset.test)));
$('cancelTest').addEventListener('click',async()=>{releaseAll();try{const r=await fetch(`${api}/test`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cancel:true})});if(!r.ok)throw Error(await r.text());await refresh()}catch(e){$('event').textContent=`取消测试失败: ${e.message}`}});
function normalizedKey(e){const c=e.code||'';if(c==='Escape')return '\u001b';if(c==='Space')return ' ';if(c.startsWith('Key')&&c.length===4)return c.slice(3).toLowerCase();if(c.startsWith('Digit')&&c.length===6)return c.slice(5);if(e.key==='Escape')return '\u001b';if(e.key==='Spacebar'||e.key===' ')return ' ';return (e.key||'').toLowerCase()}
addEventListener('keydown',e=>{const k=normalizedKey(e);if('wasd'.includes(k)){e.preventDefault();if(!e.repeat)pressHeld(k);return}if('qe123456789'.includes(k)){e.preventDefault();if(!e.repeat)sendKey(k,'press');return}if(k===' '||k==='x'||k==='z'||k==='r'||k==='\u001b'){e.preventDefault();if(!e.repeat){held.clear();document.querySelectorAll('.move.held').forEach(b=>b.classList.remove('held'));sendKey(k,'press')}}},true);
addEventListener('keyup',e=>{const k=normalizedKey(e);if('wasd'.includes(k)){e.preventDefault();releaseHeld(k)}},true);addEventListener('blur',releaseAll);document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseAll()});
addEventListener('message',e=>{if(e.source!==parent||!e.data||e.data.type!=='b2-key')return;try{const source=new URL(e.origin);if(source.hostname!=='127.0.0.1'&&source.hostname!=='localhost')return}catch{return}const k=e.data.key,action=e.data.action;if(typeof k!=='string'||k.length!==1||!['press','release'].includes(action))return;if('wasd'.includes(k)){action==='press'?pressHeld(k):releaseHeld(k);return}if(action==='press'&&'qe123456789'.includes(k)){sendKey(k,'press');return}if(action==='press'&&(k===' '||k==='x'||k==='z'||k==='r'||k==='\u001b')){held.clear();document.querySelectorAll('.move.held').forEach(b=>b.classList.remove('held'));sendKey(k,'press')}});
function arrow(ctx,x1,y1,x2,y2,color){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();const a=Math.atan2(y2-y1,x2-x1);ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-15*Math.cos(a-.5),y2-15*Math.sin(a-.5));ctx.lineTo(x2-15*Math.cos(a+.5),y2-15*Math.sin(a+.5));ctx.closePath();ctx.fill()}
function draw(){const c=$('motion'),ctx=c.getContext('2d'),w=c.width,h=c.height,cx=w/2,cy=h/2;ctx.clearRect(0,0,w,h);ctx.fillStyle='#22323d';ctx.strokeStyle='#5f7889';ctx.lineWidth=2;ctx.beginPath();ctx.roundRect(cx-70,cy-36,140,72,18);ctx.fill();ctx.stroke();ctx.fillStyle='#91a9ba';for(const sx of [-1,1])for(const sy of [-1,1])ctx.fillRect(cx+sx*58-9,cy+sy*48-14,18,28);const scale=145;arrow(ctx,cx,cy,cx+last.vy/profile.lateral_speed*scale,cy-last.vx/profile.forward_speed*scale,'#48c7e8');if(Math.abs(last.wz)>.001){ctx.strokeStyle='#ffb65c';ctx.lineWidth=6;ctx.beginPath();ctx.arc(cx,cy,86,last.wz>0?.2:Math.PI+.2,last.wz>0?Math.PI*1.45:Math.PI*2.45,last.wz<0);ctx.stroke()}}
function value(t,keys){for(const k of keys)if(t[k]!==undefined&&t[k]!==null)return t[k];return null}function metric(id,v,suffix='',digits=2){$(id).textContent=v===null||v===undefined?'--'+suffix:`${Number(v).toFixed(digits)}${suffix}`}
function renderTelemetry(t={}){const fault=Boolean(t.safety_fault_latched||t.emergency_stop_latched);$('safetyFlag').textContent=fault?'LATCHED':'NORMAL';$('safetyFlag').classList.toggle('fault-text',fault);$('batterySoc').textContent=t.battery_soc_percent===undefined?'--%':`${t.battery_soc_percent}%`;$('batteryCurrent').textContent=t.battery_current_raw??'--';$('motorTemp').textContent=t.max_motor_temperature_c===undefined?'--°C':`${t.max_motor_temperature_c}°C`;$('maxTau').textContent=t.max_abs_tau_est===undefined?'--':Number(t.max_abs_tau_est).toFixed(1);$('lostSum').textContent=t.motor_lost_sum??'--';$('crcErrors').textContent=value(t,['crc_errors','lowstate_crc_errors'])??'--';metric('ddsHz',value(t,['dds_hz','dds_frequency_hz']),' Hz',1);metric('jitterNow',value(t,['control_jitter_ms','jitter_current_ms','control_loop_jitter_ms']),' ms',3);metric('jitterMax',value(t,['control_jitter_max_ms','jitter_max_ms','max_control_loop_jitter_ms']),' ms',3);metric('policyE2e',value(t,['policy_e2e_ms','policy_latency_ms','end_to_end_latency_ms']),' ms',2);metric('cpuPercent',value(t,['cpu_percent','cpu_usage_percent']),'%',1);metric('packetLoss',value(t,['network_packet_loss_percent','packet_loss_percent']),'%',2);const rows=t.joints||[];$('jointRows').innerHTML=rows.length?rows.map(j=>`<tr><td>${j.name}</td><td>${j.target.toFixed(3)}</td><td>${j.actual.toFixed(3)}</td><td>${(j.target-j.actual).toFixed(3)}</td><td>${j.dq.toFixed(3)}</td><td>${j.tau.toFixed(1)}</td><td>${j.temperature}</td><td>${j.lost}</td></tr>`).join(''):'<tr><td colspan="8">等待 LowState</td></tr>'}
function renderFault(f={},fallback='NONE'){const text=(id,v,suffix='')=>$(id).textContent=v===undefined||v===null||v===''?'--':`${v}${suffix}`;text('faultJoint',f.joint??f.joint_name);text('faultMetric',f.metric??f.reason);text('faultValue',f.value??f.actual);text('faultThreshold',f.threshold??f.limit);text('faultDuration',f.duration_s??f.persisted_for_s,(f.duration_s??f.persisted_for_s)===undefined?'':' s');text('faultOutput',f.output_mode??f.output??fallback)}
async function refresh(){try{const r=await fetch(`${api}/status`,{cache:'no-store'}),s=await r.json();last=s.command;profile=s.command_profile;$('vx').textContent=s.command.vx.toFixed(2);$('vy').textContent=s.command.vy.toFixed(2);$('wz').textContent=s.command.wz.toFixed(2);$('forwardSpeed').textContent=`${profile.forward_speed.toFixed(2)} m/s`;$('lateralSpeed').textContent=`${profile.lateral_speed.toFixed(2)} m/s`;$('yawStep').textContent=profile.yaw_step.toFixed(2);document.querySelectorAll('.yaw-step').forEach(x=>x.textContent=profile.yaw_step.toFixed(2));document.querySelectorAll('[data-speed]').forEach(x=>x.classList.toggle('active',Number(x.dataset.speed)===Math.round(profile.forward_speed*10)));$('phase').textContent=s.phase;$('infer').textContent=`${s.inference_ms.toFixed(2)} ms`;$('inferDetail').textContent=`${s.inference_ms.toFixed(2)} ms`;$('error').textContent=`${s.tracking_error_rad.toFixed(3)} rad`;$('inputSource').textContent=s.input_source;$('sequence').textContent=s.state_sequence;$('event').textContent=s.last_event;$('modeDetail').textContent=`${s.control_mode} · ${s.mode_detail}`;$('lifecycleState').textContent=s.lifecycle_state;$('outputMode').textContent=s.output_mode;document.querySelectorAll('[data-life]').forEach(x=>x.classList.toggle('active',x.dataset.life===s.lifecycle_state));const native=$('nativeMode'),custom=$('customMode');native.disabled=!s.mode_switch_enabled||(s.mode_transition&&s.control_mode!=='CUSTOM_STARTING');custom.disabled=!s.mode_switch_enabled||s.mode_transition;native.classList.toggle('active',s.control_mode==='NATIVE');custom.classList.toggle('active',s.control_mode==='CUSTOM'||s.control_mode==='CUSTOM_STARTING');const keyboardInput=$('keyboardInput'),remoteInput=$('remoteInput'),sourceLocked=!s.input_source_switch_enabled||s.control_mode!=='NATIVE';keyboardInput.disabled=sourceLocked;remoteInput.disabled=sourceLocked;keyboardInput.classList.toggle('active',s.input_source.includes('KEYBOARD'));remoteInput.classList.toggle('active',s.input_source.includes('REMOTE'));document.querySelectorAll('[data-motion]').forEach(x=>x.disabled=!s.motion_commands_enabled);const b=$('connection');b.textContent=s.gateway_connected?'LOWSTATE 在线':'等待 LOWSTATE';b.classList.toggle('live',s.gateway_connected);renderTelemetry(s.telemetry);renderFault(s.fault||s.telemetry.fault||{},s.output_mode);draw()}catch(e){const b=$('connection');b.textContent='界面服务断开';b.classList.remove('live')}}
setInterval(refresh,100);refresh();
</script>
</body>
</html>
"""


class OperatorWebUI:
    """Serve a local control panel and expose thread-safe live status."""

    def __init__(
        self,
        command: InteractiveVelocityCommand,
        *,
        port: int = 8765,
        mode_switch_enabled: bool = False,
        input_source: str = "WEB KEYBOARD",
        native_keyboard_port: int | None = None,
        event_log_path: str | Path | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("operator UI port must be 0..65535")
        if native_keyboard_port is not None and not 0 <= native_keyboard_port <= 65535:
            raise ValueError("native keyboard port must be 0..65535")
        self.command = command
        self.port = port
        self.token = secrets.token_urlsafe(18)
        self.status = OperatorStatus(
            mode_switch_enabled=mode_switch_enabled,
            input_source=input_source,
            input_source_switch_enabled=mode_switch_enabled,
        )
        self._display_command: np.ndarray | None = None
        self._telemetry: dict[str, Any] = {}
        self._fault: dict[str, Any] = {}
        if mode_switch_enabled:
            self.status.control_mode = "UNKNOWN"
            self.status.mode_detail = "等待 MotionSwitcher 检查"
            self.status.motion_commands_enabled = False
        self._pending_mode_request: str | None = None
        self._pending_input_source_request: str | None = None
        self._pending_test_request: dict[str, Any] | None = None
        self._pending_test_cancel = False
        self._lock = threading.Lock()
        self._event_lock = threading.Lock()
        self.event_log_path = (
            None if event_log_path is None else Path(event_log_path).expanduser()
        )
        if self.event_log_path is not None:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.native_keyboard_port = native_keyboard_port
        self._native_keyboard_socket: socket.socket | None = None
        self._native_keyboard_thread: threading.Thread | None = None
        self.url: str | None = None

    def _log_event(self, event_type: str, **details: Any) -> None:
        if self.event_log_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **details,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._event_lock:
            with self.event_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line)

    def _apply_key(self, key: str, action: str, *, source: str = "") -> dict[str, Any]:
        if not isinstance(key, str) or len(key) != 1:
            raise ValueError("key must be one character")
        if action not in ("press", "release"):
            raise ValueError("action must be press or release")
        with self._lock:
            if not self.status.motion_commands_enabled:
                raise ValueError("当前模式不接受运动命令")
        event = (
            self.command.apply_key(key)
            if action == "press"
            else self.command.release_key(key)
        )
        if event is None:
            raise ValueError("unsupported key")
        label, values = event
        prefix = f"{source} " if source else ""
        with self._lock:
            self.status.last_event = (
                f"{prefix}{label}: [{values[0]:.2f}, {values[1]:.2f}, "
                f"{values[2]:.2f}]"
            )
        self._log_event(
            "keyboard",
            source=source or "WEB",
            key=key,
            action=action,
            label=label,
            command=[float(item) for item in values],
        )
        return self.snapshot()

    def _serve_native_keyboard(self, listener: socket.socket) -> None:
        while self._native_keyboard_socket is listener:
            try:
                payload, _ = listener.recvfrom(16)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(payload) != 2:
                continue
            try:
                key = payload[:1].decode("ascii")
                action = {b"p": "press", b"r": "release"}[payload[1:2]]
                self._apply_key(key, action, source="MUJOCO")
            except (KeyError, UnicodeDecodeError, ValueError):
                continue

    def open(self, *, open_browser: bool = True) -> str:
        if self._server is not None:
            assert self.url is not None
            return self.url
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == f"/{owner.token}/":
                    body = _PAGE.replace("__TOKEN__", owner.token).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == f"/api/{owner.token}/status":
                    self._json(owner.snapshot())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                if self.path not in (
                    f"/api/{owner.token}/key",
                    f"/api/{owner.token}/mode",
                    f"/api/{owner.token}/source",
                    f"/api/{owner.token}/test",
                ):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= length <= 512:
                        raise ValueError("invalid request length")
                    payload = json.loads(self.rfile.read(length))
                    if self.path == f"/api/{owner.token}/source":
                        source = payload.get("source")
                        if not owner.status.input_source_switch_enabled:
                            raise ValueError("当前控制台未启用命令源切换")
                        if source not in ("keyboard", "remote"):
                            raise ValueError("source must be keyboard or remote")
                        if payload.get("confirmed") is not True:
                            raise ValueError("source switch requires confirmation")
                        owner.command.zero()
                        with owner._lock:
                            owner._pending_input_source_request = source
                            owner.status.motion_commands_enabled = False
                            owner.status.last_event = (
                                "已请求切换命令源：" + source.upper()
                            )
                        owner._log_event("source", source=source, confirmed=True)
                        self._json(owner.snapshot())
                        return
                    if self.path == f"/api/{owner.token}/mode":
                        mode = payload.get("mode")
                        if not owner.status.mode_switch_enabled:
                            raise ValueError("当前控制台未启用硬件模式切换")
                        if mode not in ("native", "custom"):
                            raise ValueError("mode must be native or custom")
                        if payload.get("confirmed") is not True:
                            raise ValueError("mode switch requires confirmation")
                        owner.command.zero()
                        with owner._lock:
                            owner._pending_mode_request = mode.upper()
                            owner.status.mode_transition = True
                            owner.status.motion_commands_enabled = False
                            owner.status.last_event = f"已请求切换至 {mode.upper()}"
                        owner._log_event("mode", mode=mode.upper(), confirmed=True)
                        self._json(owner.snapshot())
                        return
                    if self.path == f"/api/{owner.token}/test":
                        if payload.get("cancel") is True:
                            owner.command.zero()
                            with owner._lock:
                                owner._pending_test_cancel = True
                                owner._pending_test_request = None
                                owner.status.last_event = "限时测试取消请求：ZERO/HOLD"
                            owner._log_event("test_cancel", command=[0.0, 0.0, 0.0])
                            self._json(owner.snapshot())
                            return
                        template_name = payload.get("template")
                        if payload.get("confirmed") is not True:
                            raise ValueError("test request requires confirmation")
                        if template_name not in TEST_TEMPLATES:
                            raise ValueError("unknown test template")
                        with owner._lock:
                            if not owner.status.motion_commands_enabled:
                                raise ValueError("当前状态不接受限时测试")
                            if owner._pending_test_request is not None:
                                raise ValueError("已有待执行的限时测试")
                        owner.command.zero()
                        request = dict(TEST_TEMPLATES[template_name])
                        request["template"] = template_name
                        if request["vx"] == "selected":
                            request["vx"] = float(owner.command.selected_linear_speed)
                        request["requested_at"] = datetime.now(timezone.utc).isoformat()
                        with owner._lock:
                            owner._pending_test_request = request
                            owner.status.motion_commands_enabled = False
                            owner.status.last_event = "已请求限时测试：" + request["label"]
                        owner._log_event("test", confirmed=True, **request)
                        self._json(owner.snapshot())
                        return
                    self._json(
                        owner._apply_key(
                            payload.get("key"), payload.get("action", "press")
                        )
                    )
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        actual_port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{actual_port}/{self.token}/"
        if self.native_keyboard_port is not None:
            listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listener.bind(("127.0.0.1", self.native_keyboard_port))
            listener.settimeout(0.1)
            self.native_keyboard_port = int(listener.getsockname()[1])
            self._native_keyboard_socket = listener
            self._native_keyboard_thread = threading.Thread(
                target=self._serve_native_keyboard,
                args=(listener,),
                name="b2-mujoco-keyboard",
                daemon=True,
            )
            self._native_keyboard_thread.start()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="b2-operator-ui",
            daemon=True,
        )
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url, new=1)
        return self.url

    def update(
        self,
        *,
        phase: str | None = None,
        gateway_connected: bool | None = None,
        state_sequence: int | None = None,
        inference_ms: float | None = None,
        tracking_error_rad: float | None = None,
        control_mode: str | None = None,
        mode_detail: str | None = None,
        mode_transition: bool | None = None,
        motion_commands_enabled: bool | None = None,
        display_command: np.ndarray | None = None,
        telemetry: dict[str, Any] | None = None,
        input_source: str | None = None,
        lifecycle_state: str | None = None,
        output_mode: str | None = None,
        fault: dict[str, Any] | None = None,
    ) -> None:
        mode_event: dict[str, str] = {}
        source_event: str | None = None
        with self._lock:
            if phase is not None:
                self.status.phase = phase
            if gateway_connected is not None:
                self.status.gateway_connected = gateway_connected
            if state_sequence is not None:
                self.status.state_sequence = state_sequence
            if inference_ms is not None:
                self.status.inference_ms = inference_ms
            if tracking_error_rad is not None:
                self.status.tracking_error_rad = tracking_error_rad
            if control_mode is not None:
                if control_mode != self.status.control_mode:
                    mode_event["control_mode"] = control_mode
                self.status.control_mode = control_mode
            if mode_detail is not None:
                self.status.mode_detail = mode_detail
            if mode_transition is not None:
                self.status.mode_transition = mode_transition
            if motion_commands_enabled is not None:
                self.status.motion_commands_enabled = motion_commands_enabled
            if display_command is not None:
                value = np.asarray(display_command, dtype=np.float32)
                if value.shape != (3,) or not np.all(np.isfinite(value)):
                    raise ValueError("display command must be a finite 3-vector")
                self._display_command = value.copy()
            if telemetry is not None:
                self._telemetry = dict(telemetry)
            if input_source is not None:
                if input_source != self.status.input_source:
                    source_event = input_source
                self.status.input_source = input_source
            if lifecycle_state is not None:
                if lifecycle_state != self.status.lifecycle_state:
                    mode_event["lifecycle_state"] = lifecycle_state
                self.status.lifecycle_state = lifecycle_state
            if output_mode is not None:
                if output_mode != self.status.output_mode:
                    mode_event["output_mode"] = output_mode
                self.status.output_mode = output_mode
            if fault is not None:
                self._fault = dict(fault)
        if source_event is not None:
            self._log_event("source", source=source_event, origin="status_update")
        if mode_event:
            self._log_event("mode", origin="status_update", **mode_event)

    def consume_mode_request(self) -> str | None:
        with self._lock:
            request = self._pending_mode_request
            self._pending_mode_request = None
            return request

    def consume_input_source_request(self) -> str | None:
        with self._lock:
            request = self._pending_input_source_request
            self._pending_input_source_request = None
            return request

    def consume_test_request(self) -> str | None:
        """Consume one confirmed timed-test request from the operator panel."""

        with self._lock:
            request = self._pending_test_request
            self._pending_test_request = None
        if request is not None:
            self._log_event("test_consumed", template=request["template"])
        return None if request is None else str(request["template"])

    def consume_test_cancel(self) -> bool:
        with self._lock:
            requested = self._pending_test_cancel
            self._pending_test_cancel = False
            return requested

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = asdict(self.status)
            display_command = (
                None if self._display_command is None else self._display_command.copy()
            )
            telemetry = dict(self._telemetry)
            fault = dict(self._fault)
        command = self.command.snapshot() if display_command is None else display_command
        result["command"] = {
            "vx": float(command[0]),
            "vy": float(command[1]),
            "wz": float(command[2]),
        }
        result["command_profile"] = {
            "forward_speed": float(self.command.selected_linear_speed),
            "max_forward_speed": float(self.command.max_linear_speed),
            "lateral_speed": float(self.command.max_lateral_speed),
            "yaw_step": float(self.command.yaw_step),
        }
        result["telemetry"] = telemetry
        result["fault"] = fault
        result["test_templates"] = {
            name: dict(template) for name, template in TEST_TEMPLATES.items()
        }
        return result

    def close(self) -> None:
        listener = self._native_keyboard_socket
        self._native_keyboard_socket = None
        if listener is not None:
            listener.close()
        if self._native_keyboard_thread is not None:
            self._native_keyboard_thread.join(timeout=1.0)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
        self._native_keyboard_thread = None


__all__ = ["OperatorStatus", "OperatorWebUI", "TEST_TEMPLATES"]
