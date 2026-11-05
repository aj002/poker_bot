import { expect, test, type Page } from '@playwright/test'
import { noHorizontalOverflow } from './support'

test.use({ viewport: { width: 390, height: 844 }, hasTouch: true })

const rank = (r: string) => (r === 'T' ? '10' : r)
const names: Record<string, string> = { s: 'spades', h: 'hearts', d: 'diamonds', c: 'clubs' }

async function pick(page: Page, cards: string[]) {
  for (const c of cards) {
    await page.getByRole('button', { name: rank(c[0]), exact: true }).click()
    await page.getByRole('button', { name: names[c[1]] }).click()
  }
}

async function start(page: Page) {
  await page.goto('/#/advisor')
  await expect(page.getByRole('form', { name: 'Session setup' })).toBeVisible()
  await page.getByRole('group', { name: 'Your seat' }).getByRole('button', { name: '3', exact: true }).click()
  await noHorizontalOverflow(page)
  await page.getByRole('button', { name: 'Start session' }).click()
  await expect(page.getByTestId('advisor')).toBeVisible()
}

test('worked example: enter a hand, read the advice, undo, next hand, reload', async ({ page }) => {
  await start(page)
  await expect(page.getByText('Hand 1 · Cash')).toBeVisible()
  await pick(page, ['Ah', 'Kd'])

  // Seat 1 has the button, so seat 4 acts first; seat 5 raises, the rest fold to you in the big blind.
  await expect(page.getByRole('region', { name: 'Seat 4 action' })).toBeVisible()
  await page.getByRole('region', { name: 'Seat 4 action' }).getByRole('button', { name: 'Fold' }).click()
  await page.getByRole('region', { name: 'Seat 5 action' }).getByRole('button', { name: /^Raise/ }).click()
  await page.getByLabel('Raise to', { exact: true }).fill('300')
  await page.getByRole('button', { name: 'Raise to 300' }).click()
  for (const seat of [6, 1, 2]) {
    await page.getByRole('region', { name: `Seat ${seat} action` }).getByRole('button', { name: 'Fold' }).click()
  }

  const advice = page.getByTestId('advice')
  await expect(advice).toBeVisible()
  await expect(page.getByTestId('advice-headline')).toHaveText(/^(Raise to [\d,]+ \([\d.]+ BB\)|Call 200|Fold|All-in)/)
  await expect(page.getByTestId('advice-reason')).not.toBeEmpty()
  await noHorizontalOverflow(page)

  // Mis-tap: undo the last fold, then enter it again.
  await page.getByRole('button', { name: 'Undo', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Seat 2 action' })).toBeVisible()
  await page.getByRole('region', { name: 'Seat 2 action' }).getByRole('button', { name: 'Fold' }).click()
  await expect(advice).toBeVisible()

  await page.getByRole('region', { name: 'Seat 3 action' }).getByRole('button', { name: 'Call 200' }).click()
  await expect(page.getByRole('region', { name: /^The flop/ })).toBeVisible()
  await pick(page, ['2c', '7d', '9s'])
  await expect(page.getByLabel('Board').locator('[data-card]')).toHaveCount(3)

  await page.getByRole('region', { name: 'Seat 3 action' }).getByRole('button', { name: 'Check' }).click()
  await page.getByRole('region', { name: 'Seat 5 action' }).getByRole('button', { name: /^Bet/ }).click()
  await page.getByLabel('Bet to', { exact: true }).fill('400')
  await page.getByRole('button', { name: 'Bet to 400' }).click()
  await expect(page.getByTestId('advice')).toBeVisible()
  await page.getByRole('region', { name: 'Seat 3 action' }).getByRole('button', { name: 'Fold' }).click()
  await expect(page.getByTestId('hand-result')).toContainText('Seat 5 wins')

  await page.reload()
  await expect(page.getByText('Hand 1 · Cash')).toBeVisible()
  await expect(page.getByTestId('hand-result')).toBeVisible()

  await page.getByRole('button', { name: 'Next hand', exact: true }).click()
  await expect(page.getByText('Hand 2 · Cash')).toBeVisible()
  await expect(page.getByText('button seat 2')).toBeVisible()
  await expect(page.getByTestId('adv-seat-5')).toContainText('10,350')
  await expect(page.getByTestId('adv-seat-3')).toContainText('9,650')

  await page.getByRole('button', { name: 'Undo', exact: true }).click()
  await expect(page.getByText('Hand 1 · Cash')).toBeVisible()
  await expect(page.getByTestId('hand-result')).toBeVisible()
  await noHorizontalOverflow(page)
})

test('a short session keeps working on a 360 px phone: hole cards, edit chips, tournament', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 })
  await page.goto('/#/advisor')
  await page.evaluate(() => localStorage.clear())
  await page.goto('/#/advisor')
  await page.getByRole('group', { name: 'Players' }).getByRole('button', { name: '4', exact: true }).click()
  await page.getByRole('group', { name: 'Game type' }).getByRole('button', { name: 'Tournament' }).click()
  await page.getByLabel('Starting stack').fill('2000')
  await page.getByRole('button', { name: 'Start session' }).click()
  await expect(page.getByText('Hand 1 · Tournament')).toBeVisible()
  await noHorizontalOverflow(page)

  await page.getByText('Blinds, button or chips changed?').click()
  await page.getByLabel('This hand big blind').fill('200')
  await page.getByLabel('This hand small blind').fill('100')
  await page.getByRole('button', { name: 'Apply to this hand' }).click()
  await expect(page.getByText('Blinds 100/200')).toBeVisible()

  await pick(page, ['Qs', 'Qh'])
  await expect(page.getByLabel('Your cards').locator('[data-card]')).toHaveCount(2)
  await noHorizontalOverflow(page)
  await page.getByRole('button', { name: 'Undo', exact: true }).click()
  await expect(page.getByLabel('Your two cards')).toBeVisible()

  await page.getByText('Session', { exact: true }).click()
  await page.getByRole('button', { name: 'End session and start a new one' }).click()
  await expect(page.getByRole('form', { name: 'Session setup' })).toBeVisible()
})
